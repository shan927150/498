"""MoEQuant EBSS score/sign correctness audit (no calibration-quality experiment).

Stages (--stage):
  synthetic   fixed counterexamples, cross-parent sign trap, tradeoff threshold, beam-width ranking, deferred-sigma
              and counterfactual tests, metamorphic tests, and a full trace audit of generate() on synthetic backends
  historical  audit of the stored 16 x 512 pilot artifacts (cumulative log-probs, routing counts, sigma, final
              scores, final selection, recorded top-8 step logs, B1 replay NLL)
  production  opt-in capture of the exact production pruning state, 4 x 32 tokens per method on the real BF16
              model, every pruning decision / propagation / final selection compared against the oracle, plus a
              teacher-forced replay of the captured sequences (GPU; ~5 min of model loading)
  report      decision.json + summary.md
The oracle lives in runs_ebss_score_audit/independent_oracle.py (NumPy float64; no production imports).
"""
from __future__ import annotations

import argparse, importlib.util, json, math, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_calib import ebss_generator as G              # noqa: E402

OUT = os.path.join(ROOT, "runs_ebss_score_audit")
PILOT = os.path.join(ROOT, "runs_ebss_generation_pilot")
TAU, W = G.TAU_REF, G.W_REF
TOL_ALG = 1e-12          # float64 algebra identities
TOL_SCORE = 1e-9         # production vs oracle score given the same per-token log-probabilities
TOL_LSM = 1e-5           # fp32 production log-softmax vs fp64 oracle log-softmax
TIE_PROD = 1e-9
TIE_FP64 = 2e-5


def load_oracle():
    spec = importlib.util.spec_from_file_location("independent_oracle", os.path.join(OUT, "independent_oracle.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


O = load_oracle()


def p(*a):
    return os.path.join(OUT, *a)


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


def prod_scores(L, logp, sigma, l, tau=TAU, use_balance=True):
    """Production Eq. 13 (moe_calib.ebss_generator.candidate_scores) on explicit float64 inputs."""
    return G.candidate_scores(torch.tensor(L, dtype=torch.float64), torch.tensor(logp, dtype=torch.float64),
                              torch.tensor(sigma, dtype=torch.float64), l, tau, use_balance).numpy()


def prod_select(scores, w):
    pp, tt, ss = G.select_best(torch.tensor(scores, dtype=torch.float64), w)
    return [(int(a), int(b), float(c)) for a, b, c in zip(pp, tt, ss)]


def kept_set(x):
    return sorted((a, b) for a, b, _ in x)


# =================================================================== fixed counterexamples
def fixed_examples():
    ex = []

    def cand(name, parent, L_hist_tokens, p_v, l, sigma):
        L = O.cumulative_logp(L_hist_tokens)
        N = -L
        r = dict(name=name, parent=parent, prefix_token_logps=list(L_hist_tokens), L_S=L, N_S=N, p_v=p_v,
                 prefix_len=l, sigma=sigma, tau=TAU,
                 J_logp=O.score_logp(L, p_v, l, sigma, TAU), J_nll=O.score_nll(N, p_v, l, sigma, TAU),
                 J_wrong_nll_sign=O.wrong_score_nll_sign(N, p_v, l, sigma, TAU),
                 J_production=float(prod_scores([L], [[p_v]], [sigma], l)[0, 0]))
        r["logp_vs_nll_abs"] = abs(r["J_logp"] - r["J_nll"])
        r["prod_vs_oracle_abs"] = abs(r["J_production"] - r["J_logp"])
        return r

    def pair(title, A, B, expect_A_better, note=""):
        ok_oracle = (A["J_logp"] < B["J_logp"]) == expect_A_better
        ok_nll = (A["J_nll"] < B["J_nll"]) == expect_A_better
        ok_prod = (A["J_production"] < B["J_production"]) == expect_A_better
        wrong = (A["J_wrong_nll_sign"] < B["J_wrong_nll_sign"]) == expect_A_better
        ex.append(dict(test=title, note=note, A=A, B=B, expected="A ranks above B" if expect_A_better else
                       "B ranks above A", oracle_logp_ok=ok_oracle, oracle_nll_ok=ok_nll, production_ok=ok_prod,
                       wrong_nll_sign_formula_gives_expected=wrong,
                       identity_ok=max(A["logp_vs_nll_abs"], B["logp_vs_nll_abs"]) <= TOL_ALG,
                       production_matches_oracle=max(A["prod_vs_oracle_abs"], B["prod_vs_oracle_abs"]) <= TOL_SCORE,
                       passed=ok_oracle and ok_nll and ok_prod))

    h = [math.log(0.5), math.log(0.4)]                                   # shared 3-token history (first token = 0)
    pair("6_high_probability_same_parent",
         cand("A p=0.8", 0, h, math.log(0.8), 3, 0.1), cand("B p=0.2", 0, h, math.log(0.2), 3, 0.1), True,
         "identical parent, history, sigma, length")
    pair("7_balance_monotonicity",
         cand("A sigma=0.1", 0, h, math.log(0.5), 3, 0.1), cand("B sigma=0.3", 1, h, math.log(0.5), 3, 0.3), True,
         "identical history logP, next-token logP and length; sigma differs")
    pair("8_cross_parent_sign_trap",
         cand("A L=-0.2", 0, [-0.2], math.log(0.5), 4, 0.1), cand("B L=-2.0", 1, [-2.0], math.log(0.5), 4, 0.1),
         True, "equal lengths (|S|=4), equal next-token p=0.5, equal sigma=0.1; only the parent history differs")
    pair("5a_one_token_prefix", cand("A p=0.6", 0, [], math.log(0.6), 1, 0.2),
         cand("B p=0.3", 0, [], math.log(0.3), 1, 0.2), True, "|S| = 1: only the unconditioned first token")
    pair("5b_very_low_probability", cand("A p=1e-3", 0, h, math.log(1e-3), 3, 0.1),
         cand("B p=1e-30", 0, h, math.log(1e-30), 3, 0.1), True, "log p = -69.08 stays finite and ordered")
    hA = [math.log(0.9), math.log(0.9), math.log(0.9)]
    hB = [math.log(0.9), math.log(0.2), math.log(0.9)]
    pair("5c_multi_token_different_histories", cand("A", 0, hA, math.log(0.3), 4, 0.15),
         cand("B", 1, hB, math.log(0.3), 4, 0.15), True, "4-token prefixes; B has one low-probability token")
    pair("5d_different_parent_sigma_same_history", cand("A", 0, hA, math.log(0.3), 4, 0.05),
         cand("B", 1, hA, math.log(0.3), 4, 0.25), True, "")
    # equal probability and equal sigma -> exact tie, broken by (parent, token)
    s = prod_scores([-1.0, -1.0], [[math.log(0.5), math.log(0.5)], [math.log(0.5), math.log(0.5)]], [0.1, 0.1], 3)
    sel = prod_select(s, 3)
    orc = O.rank_global(O.candidate_matrix([-1.0, -1.0], np.log([[0.5, 0.5], [0.5, 0.5]]), [0.1, 0.1], 3, TAU), 3)
    ex.append(dict(test="5e_exact_tie_equal_probability_equal_sigma", scores=s.tolist(), production_kept=sel,
                   oracle_kept=orc, expected=[[0, 0], [0, 1], [1, 0]],
                   passed=kept_set(sel) == kept_set(orc) == [(0, 0), (0, 1), (1, 0)]
                   and [x[:2] for x in sel] == [(0, 0), (0, 1), (1, 0)]))
    return ex


def cross_parent_trap():
    L_A, L_B, pv, sig, l = -0.2, -2.0, math.log(0.5), 0.1, 4
    N_A, N_B = -L_A, -L_B
    rows = {}
    for name, f in [("correct_logp", lambda L, N: O.score_logp(L, pv, l, sig, TAU)),
                    ("correct_nll", lambda L, N: O.score_nll(N, pv, l, sig, TAU)),
                    ("wrong_minus_nll_plus_logp", lambda L, N: O.wrong_score_nll_sign(N, pv, l, sig, TAU)),
                    ("production_candidate_scores", lambda L, N: float(prod_scores([L], [[pv]], [sig], l)[0, 0]))]:
        a, b = f(L_A, N_A), f(L_B, N_B)
        rows[name] = dict(score_A=a, score_B=b, prefers="A" if a < b else ("B" if b < a else "tie"))
    # the same trap through the full production search (generate()) with a scripted backend
    e2e = trap_end_to_end()
    return dict(setup=dict(L_A=L_A, L_B=L_B, N_A=N_A, N_B=N_B, next_token_p=0.5, p_v=pv, sigma=sig, tau=TAU,
                           prefix_len=l, new_length=l + 1),
                hand_calculation=dict(
                    correct=f"A: (0.2 + 0.693147)/5 + 0.1/1.3 = {(0.2 - pv) / 5 + sig / TAU:.6f}; "
                            f"B: (2.0 + 0.693147)/5 + 0.1/1.3 = {(2.0 - pv) / 5 + sig / TAU:.6f}",
                    wrong=f"A: -(0.2 - 0.693147)/5 + 0.0769 = {-(0.2 + pv) / 5 + sig / TAU:.6f}; "
                          f"B: -(2.0 - 0.693147)/5 + 0.0769 = {-(2.0 + pv) / 5 + sig / TAU:.6f}"),
                formulas=rows, end_to_end_generate=e2e,
                passed=(rows["correct_logp"]["prefers"] == rows["correct_nll"]["prefers"] ==
                        rows["production_candidate_scores"]["prefers"] == "A"
                        and rows["wrong_minus_nll_plus_logp"]["prefers"] == "B" and e2e["passed"]))


# =================================================================== scripted backends (production generate())
class ScriptedBackend(G.Backend):
    """Logits and routing are explicit functions of the full prefix (tuple of ids)."""

    def __init__(self, V, L, E, k, logits_fn, route_fn):
        self.vocab_limit, self.n_layers, self.n_experts, self.top_k = V, L, E, k
        self.logits_fn, self.route_fn = logits_fn, route_fn
        self.prefixes = []

    def _out(self):
        lg = torch.stack([torch.tensor(self.logits_fn(pf), dtype=torch.float32) for pf in self.prefixes])
        sel = torch.stack([torch.tensor(self.route_fn(pf), dtype=torch.long) for pf in self.prefixes])
        return lg, sel

    def start(self, tokens):
        self.prefixes = [(int(t),) for t in tokens]; return self._out()

    def step(self, tokens):
        self.prefixes = [pf + (int(t),) for pf, t in zip(self.prefixes, tokens)]; return self._out()

    def reorder(self, rows):
        self.prefixes = [self.prefixes[int(r)] for r in rows]


def trap_end_to_end():
    """V = 4, w = 2, seqlen = 3. Step 1 from the first token (id 3): log P = [-0.2, -2.0, rest].
    Step 2: every token log P = log 0.25 for both parents; identical routing -> identical sigma.
    Correct minimisation keeps both children of parent A (token 0); the wrong-sign variant would keep B's."""
    rest = math.log((1 - math.exp(-0.2) - math.exp(-2.0)) / 2)

    def logits_fn(pf):
        return [-0.2, -2.0, rest, rest] if len(pf) == 1 else [math.log(0.25)] * 4

    def route_fn(pf):
        return [[0, 1]]                                  # same routing for every token -> equal sigma
    out = {}
    for ub in (True, False):
        rec = []
        r = G.generate(ScriptedBackend(4, 1, 4, 2, logits_fn, route_fn), [3], seqlen=3, w=2, tau=TAU,
                       use_balance=ub, audit=rec.append)
        st2 = [x for x in rec if x["kind"] == "step" and x["step"] == 2][0]
        kept_par = [int(st2["tokens"][int(q)][1]) for q in st2["parent"]]       # parent identified by its token
        out["EBSS" if ub else "PO"] = dict(parents_cum_logp=st2["cum"].tolist(), kept_children_parent_token=kept_par,
                                           chosen_sentence=r.tokens[0].tolist(), cum_logp=float(r.cum_logp[0]))
    out["passed"] = all(v["kept_children_parent_token"] == [0, 0] and v["chosen_sentence"][:2] == [3, 0]
                        for v in out.values())
    return out


def deferred_sigma_end_to_end():
    """Two parents with EQUAL cumulative log P but different routing balance; children all equiprobable.
    First token 4 routes to experts {0,1}. Step 1: tokens 1 and 2 both p = 0.4. Token 1 routes {0,1} (same as the
    first token -> concentrated), token 2 routes {2,3} (complementary -> balanced). After pruning parent 0 = token 1
    (lower id on the tie), parent 1 = token 2. EBSS must keep both children of parent 1 (lower sigma);
    Probability-Only (exact tie) keeps parent 0's; reversed sigma would keep parent 0's as well."""
    def logits_fn(pf):
        return [math.log(0.1), math.log(0.4), math.log(0.4), math.log(0.05), math.log(0.05)] if len(pf) == 1 \
            else [math.log(0.2)] * 5

    def route_fn(pf):
        last = pf[-1]
        return [[2, 3]] if last == 2 else [[0, 1]]
    res = {}
    for ub in (True, False):
        rec = []
        G.generate(ScriptedBackend(5, 1, 4, 2, logits_fn, route_fn), [4], seqlen=3, w=2, tau=TAU, use_balance=ub,
                   audit=rec.append)
        st2 = [x for x in rec if x["kind"] == "step" and x["step"] == 2][0]
        sig = st2["sigma"].tolist()
        oracle_sig = [O.sigma_eq4(O.counts_from_routing(st2["hist"][b].numpy(), 4), 2) for b in range(2)]
        kept_parent_tokens = [int(st2["tokens"][int(q)][1]) for q in st2["parent"]]
        orc = O.rank_global(O.candidate_matrix(st2["cum"].numpy(), O.log_softmax64(st2["logits"].numpy()),
                                               oracle_sig, 2, TAU, ub), 2)
        orc_rev = O.rank_global(O.candidate_matrix(st2["cum"].numpy(), O.log_softmax64(st2["logits"].numpy()),
                                                   oracle_sig, 2, TAU, True, "wrong_reversed_sigma"), 2)
        res["EBSS" if ub else "PO"] = dict(
            parent_tokens=[int(t[1]) for t in st2["tokens"]], parent_cum_logp=st2["cum"].tolist(),
            parent_sigma_production=sig, parent_sigma_oracle=oracle_sig,
            kept_children_of_parent_token=kept_parent_tokens,
            oracle_kept=[(a, b) for a, b, _ in orc], reversed_sigma_kept=[(a, b) for a, b, _ in orc_rev],
            production_kept=[(int(a), int(b)) for a, b in zip(st2["parent"], st2["token"])])
    e, po = res["EBSS"], res["PO"]
    res["passed"] = (e["parent_sigma_production"][1] < e["parent_sigma_production"][0]
                     and max(abs(a - b) for a, b in zip(e["parent_sigma_production"], e["parent_sigma_oracle"])) < 1e-12
                     and e["kept_children_of_parent_token"] == [2, 2] and po["kept_children_of_parent_token"] == [1, 1]
                     and sorted(e["production_kept"]) == sorted(e["oracle_kept"])
                     and all(a == 0 for a, _ in e["reversed_sigma_kept"]))
    return res


# =================================================================== tradeoff threshold
def tradeoff():
    """A: L = -1.0, sigma 0.30;  B: L = -1.5, sigma 0.10; same next-token p = 0.5, |S| = 4.
    Probability advantage of A = (1.5 - 1.0)/5 = 0.1; balance advantage of B = 0.2/tau.
    B wins iff 0.2/tau > 0.1  <=>  tau < 2 (balance weight 1/tau > 0.5). At tau = 2 exact tie -> parent A (index 0)."""
    pv, l = math.log(0.5), 4
    rows = []
    for tau in (0.5, 1.0, 1.3, 1.9, 2.0, 2.1, 3.0, 10.0, 1e9):
        Sp = prod_scores([-1.0, -1.5], [[pv], [pv]], [0.30, 0.10], l, tau)
        orc = O.candidate_matrix([-1.0, -1.5], [[pv], [pv]], [0.30, 0.10], l, tau)
        sel_p = prod_select(Sp, 1)[0][0]
        sel_o = O.rank_global(orc, 1)[0][0]
        expected = 1 if tau < 2.0 else 0
        rows.append(dict(tau=tau, balance_weight=1 / tau, score_A=float(Sp[0, 0]), score_B=float(Sp[1, 0]),
                         oracle_A=float(orc[0, 0]), oracle_B=float(orc[1, 0]), production_winner="AB"[sel_p],
                         oracle_winner="AB"[sel_o], expected_winner="AB"[expected],
                         ok=sel_p == sel_o == expected))
    wrong_mult = O.candidate_matrix([-1.0, -1.5], [[pv], [pv]], [0.30, 0.10], l, 3.0, True, "wrong_tau_multiplies")
    return dict(setup="A: L=-1.0 sigma=0.30; B: L=-1.5 sigma=0.10; p_v=log 0.5; |S|=4",
                threshold=dict(tau_star=2.0, balance_weight_star=0.5,
                               derivation="(1.5-1.0)/5 = 0.1 = (0.30-0.10)/tau  ->  tau* = 2"),
                rows=rows, tau_as_multiplier_control=dict(tau=3.0, A=float(wrong_mult[0, 0]), B=float(wrong_mult[1, 0]),
                                                          winner="B" if wrong_mult[1, 0] < wrong_mult[0, 0] else "A",
                                                          note="multiplying by tau would make B win at tau=3"),
                passed=all(r["ok"] for r in rows))


# =================================================================== beam-width ranking
def beam_width_ranking():
    """3 parents x V = 5, explicit values chosen so that every wrong variant changes the kept set."""
    Lp = np.array([-0.5, -1.4, -0.9])
    probs = np.array([[0.50, 0.20, 0.15, 0.10, 0.05],
                      [0.60, 0.25, 0.08, 0.05, 0.02],
                      [0.40, 0.30, 0.20, 0.06, 0.04]])
    lp = np.log(probs)
    sig = np.array([0.40, 0.05, 0.20])
    l = 6
    out = dict(parents_L=Lp.tolist(), next_token_probs=probs.tolist(), parent_sigma=sig.tolist(), prefix_len=l,
               tau=TAU, cases=[])
    ok = True
    for w in (2, 4):
        for ub in (True, False):
            S = prod_scores(Lp, lp, sig, l, TAU, ub)
            orcS = O.candidate_matrix(Lp, lp, sig, l, TAU, ub)
            prod = prod_select(S, w)
            orc = O.rank_global(orcS, w)
            variants = {
                "maximise": O.rank_global(orcS, w, maximise=True),
                "per_parent_topk": O.wrong_rank_per_parent(orcS, w),
                "wrong_nll_sign": O.rank_global(O.candidate_matrix(Lp, lp, sig, l, TAU, ub, "wrong_nll_sign"), w),
                "probability_only": O.rank_global(O.candidate_matrix(Lp, lp, sig, l, TAU, False), w),
                "balance_after_pruning": sorted(O.rank_global(O.candidate_matrix(Lp, lp, sig, l, TAU, False), w),
                                                key=lambda x: x[2] + sig[x[0]] / TAU),
                "wrong_parent_sigma(rolled)": O.rank_global(O.candidate_matrix(Lp, lp, np.roll(sig, 1), l, TAU, ub), w),
                "wrong_parent_history(rolled)": O.rank_global(O.candidate_matrix(np.roll(Lp, 1), lp, sig, l, TAU, ub),
                                                              w),
                "reversed_sigma": O.rank_global(O.candidate_matrix(Lp, lp, sig, l, TAU, ub, "wrong_reversed_sigma"), w),
            }
            vk = {k: [x[:2] for x in v] for k, v in variants.items()}
            distinguishes = {k: kept_set(v) != kept_set(orc) for k, v in variants.items()}
            match = [x[:2] for x in prod] == [x[:2] for x in orc] and \
                max(abs(a[2] - b[2]) for a, b in zip(prod, orc)) <= TOL_SCORE
            ok &= match
            out["cases"].append(dict(w=w, use_balance=ub, production_kept=[x[:2] for x in prod],
                                     production_scores=[x[2] for x in prod], oracle_kept=[x[:2] for x in orc],
                                     oracle_scores=[x[2] for x in orc], production_matches_oracle_in_order=match,
                                     wrong_variant_kept=vk, wrong_variant_differs_from_correct=distinguishes))
    ok &= stable_tie_case()
    out["stable_tie_breaking"] = "exact ties ordered by (parent, token): see 5e and the tie block below"
    out["passed"] = bool(ok)
    return out


def stable_tie_case():
    S = np.zeros((3, 4)); S[2, 0] = -1.0
    return [x[:2] for x in prod_select(S, 4)] == [(2, 0), (0, 0), (0, 1), (0, 2)] == \
        [x[:2] for x in O.rank_global(S, 4)]


# =================================================================== counterfactual audit of one frozen pool
def counterfactual(Lp, lp, sig, l, w, tau=TAU):
    S = {"probability_only": O.candidate_matrix(Lp, lp, sig, l, tau, False),
         "ebss": O.candidate_matrix(Lp, lp, sig, l, tau, True),
         "ebss_sigma_disabled": O.candidate_matrix(Lp, lp, np.zeros_like(sig), l, tau, True),
         "ebss_reversed_sigma": O.candidate_matrix(Lp, lp, sig, l, tau, True, "wrong_reversed_sigma")}
    kept = {k: O.rank_global(v, w) for k, v in S.items()}

    def mean_sigma(x):
        return float(np.mean([sig[a] for a, _, _ in x]))
    return dict(kept={k: [x[:2] for x in v] for k, v in kept.items()},
                mean_parent_sigma_of_kept={k: mean_sigma(v) for k, v in kept.items()},
                disabled_equals_po=kept_set(kept["ebss_sigma_disabled"]) == kept_set(kept["probability_only"]),
                ebss_le_po_sigma=mean_sigma(kept["ebss"]) <= mean_sigma(kept["probability_only"]) + 1e-15,
                reversed_ge_po_sigma=mean_sigma(kept["ebss_reversed_sigma"]) >= mean_sigma(kept["probability_only"]) - 1e-15)


# =================================================================== metamorphic tests
def metamorphic(n_sets=300, seed=20261110):
    rng = np.random.default_rng(seed)
    names = ["logp_up_improves", "history_up_improves", "sigma_up_worsens", "sigma0_equals_po", "tau_inf_equals_po",
             "tau_up_shrinks_balance_spread", "constant_shift_invariant", "enumeration_order_invariant",
             "production_equals_oracle", "logp_nll_identity", "counterfactual_ordering"]
    counts = {n: [0, 0] for n in names}
    fails = []
    power = {v: 0 for v in ("wrong_nll_sign", "maximise", "per_parent_topk", "reversed_sigma", "probability_only")}
    n_cf = 0

    def rec(name, ok, ex):
        counts[name][0] += 1
        if not ok:
            counts[name][1] += 1
            if len(fails) < 50:
                fails.append(dict(test=name, **ex))
    for it in range(n_sets):
        B = int(rng.integers(1, 5)); V = int(rng.integers(3, 40)); w = int(rng.integers(1, min(B * V, 6) + 1))
        l = int(rng.integers(1, 600)); tau = float(rng.choice([0.5, 1.2, 1.3, 2.0]))
        Lp = -rng.gamma(2.0, 0.6, size=B) * l / 5
        lp = O.log_softmax64(rng.normal(0, 3, size=(B, V)))
        sig = rng.uniform(0, 0.3, size=B)
        ex = dict(set=it, B=B, V=V, w=w, prefix_len=l, tau=tau, L=Lp.tolist(), sigma=sig.tolist())
        S = prod_scores(Lp, lp, sig, l, tau)
        orc = O.candidate_matrix(Lp, lp, sig, l, tau)
        rec("production_equals_oracle", np.abs(S - orc).max() <= TOL_SCORE and
            kept_set(prod_select(S, w)) == kept_set(O.rank_global(orc, w)), ex)
        nll = np.array([[O.score_nll(-Lp[b], lp[b, v], l, sig[b], tau) for v in range(V)] for b in range(B)])
        lgp = np.array([[O.score_logp(Lp[b], lp[b, v], l, sig[b], tau) for v in range(V)] for b in range(B)])
        rec("logp_nll_identity", np.abs(nll - lgp).max() <= TOL_ALG, ex)
        b, v = int(rng.integers(B)), int(rng.integers(V))
        lp2 = lp.copy(); lp2[b, v] += 0.1 * abs(lp[b, v]) + 1e-3
        rec("logp_up_improves", prod_scores(Lp, lp2, sig, l, tau)[b, v] < S[b, v], ex)
        Lp2 = Lp.copy(); Lp2[b] += 0.5
        rec("history_up_improves", bool(np.all(prod_scores(Lp2, lp, sig, l, tau)[b] < S[b])), ex)
        sig2 = sig.copy(); sig2[b] += 0.05
        rec("sigma_up_worsens", bool(np.all(prod_scores(Lp, lp, sig2, l, tau)[b] > S[b])), ex)
        Spo = prod_scores(Lp, lp, sig, l, tau, False)
        rec("sigma0_equals_po", np.array_equal(prod_scores(Lp, lp, np.zeros(B), l, tau), Spo)
            and prod_select(prod_scores(Lp, lp, np.zeros(B), l, tau), w) == prod_select(Spo, w), ex)
        # tau -> infinity: ranking equals PO unless two PO scores are closer than the residual balance spread
        Sinf = prod_scores(Lp, lp, sig, l, 1e15)
        rec("tau_inf_equals_po", kept_set(prod_select(Sinf, w)) == kept_set(prod_select(Spo, w)), ex)
        spreads = [np.ptp(prod_scores(Lp, lp, sig, l, t) - Spo) for t in (0.5, 1.0, 2.0, 8.0)]
        rec("tau_up_shrinks_balance_spread", all(a >= b - 1e-15 for a, b in zip(spreads, spreads[1:])), ex)
        rec("constant_shift_invariant", kept_set(prod_select(S + 3.7, w)) == kept_set(prod_select(S, w)), ex)
        # enumeration order: permute parents and tokens, map back; skip sets with exact/near ties at the cut
        flat = np.sort(S.reshape(-1))
        if w < flat.size and flat[w] - flat[w - 1] > 1e-12:
            pp, pv = rng.permutation(B), rng.permutation(V)
            Sperm = prod_scores(Lp[pp], lp[pp][:, pv], sig[pp], l, tau)
            back = sorted((int(pp[a]), int(pv[t])) for a, t, _ in prod_select(Sperm, w))
            rec("enumeration_order_invariant", back == kept_set(prod_select(S, w)), ex)
        if B > 1:
            cf = counterfactual(Lp, lp, sig, l, w, tau); n_cf += 1
            rec("counterfactual_ordering", cf["disabled_equals_po"] and cf["ebss_le_po_sigma"]
                and cf["reversed_ge_po_sigma"], ex)
            corr = kept_set(O.rank_global(orc, w))
            power["wrong_nll_sign"] += kept_set(O.rank_global(O.candidate_matrix(Lp, lp, sig, l, tau, True,
                                                                                 "wrong_nll_sign"), w)) != corr
            power["maximise"] += kept_set(O.rank_global(orc, w, True)) != corr
            power["per_parent_topk"] += kept_set(O.wrong_rank_per_parent(orc, w)) != corr
            power["reversed_sigma"] += kept_set(O.rank_global(O.candidate_matrix(Lp, lp, sig, l, tau, True,
                                                                                 "wrong_reversed_sigma"), w)) != corr
            power["probability_only"] += kept_set(O.rank_global(O.candidate_matrix(Lp, lp, sig, l, tau, False),
                                                                w)) != corr
    return dict(seed=seed, n_sets=n_sets,
                results={k: dict(checked=v[0], failed=v[1]) for k, v in counts.items()},
                wrong_variant_detection=dict(n_multi_parent_sets=n_cf,
                                             sets_where_variant_changes_kept_set=power),
                failing_examples=fails, passed=all(v[1] == 0 for v in counts.values()))


# =================================================================== trace auditor (production generate())
class TraceAuditor:
    """Consumes the opt-in audit records of generate() one step at a time and checks every quantity and every
    pruning decision against the oracle. Keeps only the previous step for propagation checks."""

    def __init__(self, n_experts, keep_examples=12):
        self.E = n_experts
        self.prev = None
        self.steps = []
        self.final = None
        self.examples = []
        self.keep = keep_examples
        self.mism = []

    def __call__(self, r):
        if r["kind"] == "final":
            return self._final(r)
        j, l, B, w, tau, ub = (r[k] for k in ("step", "prefix_len", "n_branches", "w", "tau", "use_balance"))
        R = r["cum"].shape[0]; N = R // B
        cum = r["cum"].numpy(); tlp = r["token_logp"].numpy(); sig = r["sigma"].numpy()
        logp_prod = r["logp"].numpy().astype(np.float64)
        logp64 = O.log_softmax64(r["logits"].numpy())
        prod_sc = r["scores"].numpy().reshape(R, -1)
        V = prod_sc.shape[1]
        d = dict(step=j, prefix_len=l, n_branches=B)
        # state
        d["prefix_length_ok"] = bool(r["tokens"].shape[1] == l and tlp.shape[1] == l - 1 and r["hist"].shape[1] == l)
        d["cum_vs_fsum_max_abs"] = float(max(abs(cum[i] - O.cumulative_logp(tlp[i])) for i in range(R)))
        oc = [O.counts_from_routing(r["hist"][i].numpy(), self.E) for i in range(R)]
        d["counts_recount_exact"] = all(np.array_equal(oc[i], r["counts"][i].numpy()) for i in range(R))
        osig = np.array([O.sigma_eq4(oc[i], l) for i in range(R)])
        d["sigma_max_abs"] = float(np.abs(osig - sig).max())
        d["sigma_nonneg"] = bool((sig >= 0).all())
        d["logsoftmax_fp64_max_abs"] = float(np.abs(logp64 - logp_prod).max())
        d["logp_nonpos"] = bool((logp_prod <= 0).all())
        # scores: oracle with the production per-token logP (pure formula check) and with fp64 logP from logits
        orc = O.candidate_matrix(cum, logp_prod, osig, l, tau, ub)
        orc64 = O.candidate_matrix(cum, logp64, osig, l, tau, ub)
        d["score_max_abs_vs_oracle"] = float(np.abs(orc - prod_sc).max())
        d["score_max_abs_vs_fp64_oracle"] = float(np.abs(orc64 - prod_sc).max())
        d["prob_component_nonneg"] = bool(((-(cum[:, None] + logp_prod)) / (l + 1) >= 0).all())
        par = r["parent"].numpy(); tok = r["token"].numpy()
        dec_ok = dec64_ok = True; n_ebss_changes = 0; gaps = []
        for i in range(N):
            rows = slice(i * B, (i + 1) * B)
            pk = sorted(zip((par[i * w:(i + 1) * w] - i * B).tolist(), tok[i * w:(i + 1) * w].tolist()))
            ok_ = O.rank_global(orc[rows], w); ok64 = O.rank_global(orc64[rows], w)
            gaps.append(O.kept_gap(orc[rows], ok_, w))
            if kept_set(ok_) != pk:
                g = O.kept_gap(orc[rows], ok_, w)
                if g > TIE_PROD:
                    dec_ok = False
                    self.mism.append(dict(step=j, sentence=i, production=pk, oracle=[x[:2] for x in ok_], gap=g))
            if kept_set(ok64) != pk and O.kept_gap(orc64[rows], ok64, w) > TIE_FP64:
                dec64_ok = False
                self.mism.append(dict(step=j, sentence=i, kind="fp64", production=pk, oracle=[x[:2] for x in ok64]))
            if ub and B > 1:
                po = O.rank_global(O.candidate_matrix(cum[rows], logp_prod[rows], osig[rows], l, tau, False), w)
                n_ebss_changes += kept_set(po) != kept_set(ok_)
                if kept_set(po) != kept_set(ok_) and len(self.examples) < self.keep:
                    self.examples.append(self._example(r, i, B, w, orc, ok_, po, osig, cum, logp_prod, l, tau,
                                                       "EBSS changes the probability-only kept set"))
            if len(self.examples) < self.keep and (j in (1, 2) or gaps[-1] < 1e-3) and i == 0:
                self.examples.append(self._example(r, i, B, w, orc, ok_, None, osig, cum, logp_prod, l, tau,
                                                   "early step" if j in (1, 2) else "near tie at the cut"))
        d["decisions_match_oracle"] = dec_ok
        d["decisions_match_fp64_oracle"] = dec64_ok
        d["min_cut_gap"] = float(min(gaps))
        d["sentences_where_ebss_changed_po_set"] = int(n_ebss_changes)
        d["n_parents_distinct_in_kept"] = int(np.mean([len(set(par[i * w:(i + 1) * w])) for i in range(N)]))
        # propagation from the previous step's decision
        if self.prev is not None:
            pv = self.prev
            pp = pv["parent"]; pt = pv["token"]
            d["tokens_propagated"] = bool(torch.equal(r["tokens"], torch.cat([pv["tokens"][pp], pt[:, None]], 1)))
            exp_cum = pv["cum"][pp].numpy() + pv["logp"][pp, pt].double().numpy()
            d["cum_propagated_max_abs"] = float(np.abs(exp_cum - cum).max())
            d["hist_prefix_propagated"] = bool(torch.equal(r["hist"][:, :-1], pv["hist"][pp]))
            d["counts_propagated"] = all(
                np.array_equal(r["counts"][q].numpy() - pv["counts"][pp[q]].numpy(),
                               O.counts_from_routing(r["hist"][q, -1:].numpy(), self.E)) for q in range(R))
            d["token_logp_propagated"] = bool(torch.equal(r["token_logp"],
                                                          torch.cat([pv["token_logp"][pp],
                                                                     pv["logp"][pp, pt].double()[:, None]], 1)))
        self.prev = dict(tokens=r["tokens"], parent=r["parent"], token=r["token"], cum=r["cum"], logp=r["logp"],
                         hist=r["hist"], counts=r["counts"], token_logp=r["token_logp"])
        self.steps.append(d)

    def _example(self, r, i, B, w, orc, ok_, po, osig, cum, logp, l, tau, why):
        rows = slice(i * B, (i + 1) * B)
        top = O.rank_global(orc[rows], min(2 * w, orc[rows].size))
        prod_sc = r["scores"].numpy().reshape(cum.shape[0], -1)[rows]
        return dict(why=why, step=r["step"], sentence=i, prefix_len=l, tau=tau, use_balance=r["use_balance"],
                    candidates=[dict(rank=k, parent=a, token=t, R_parent=float(cum[i * B + a]),
                                     logp_v=float(logp[i * B + a, t]),
                                     prob_component=float(-(cum[i * B + a] + logp[i * B + a, t]) / (l + 1)),
                                     sigma_parent=float(osig[i * B + a]), balance=float(osig[i * B + a] / tau),
                                     oracle_score=s, production_score=float(prod_sc[a, t]), kept_by_oracle=k < w)
                                for k, (a, t, s) in enumerate(top)],
                    production_kept=sorted(zip((r["parent"].numpy()[i * w:(i + 1) * w] - i * B).tolist(),
                                               r["token"].numpy()[i * w:(i + 1) * w].tolist())),
                    probability_only_kept=None if po is None else [x[:2] for x in po])

    def _final(self, r):
        B = r["final_scores"].shape[1]
        cum = r["cum"].numpy(); n = r["length"]
        oc = [O.counts_from_routing(r["hist"][q].numpy(), self.E) for q in range(cum.shape[0])]
        osig = [O.sigma_eq4(c, n) for c in oc]
        fs = np.array([O.final_score(O.cumulative_logp(r["token_logp"][q].numpy()), n, osig[q], r["tau"],
                                     r["use_balance"]) for q in range(cum.shape[0])]).reshape(-1, B)
        pick = [O.final_select(cum[i * B:(i + 1) * B], osig[i * B:(i + 1) * B], n, r["tau"], r["use_balance"])
                for i in range(fs.shape[0])]
        self.final = dict(final_score_max_abs=float(np.abs(fs - r["final_scores"].numpy()).max()),
                          pick_production=r["pick"].tolist(), pick_oracle=pick,
                          pick_matches=r["pick"].tolist() == pick,
                          counts_exact=all(np.array_equal(oc[q], r["counts"][q].numpy()) for q in range(len(oc))),
                          length_ok=r["tokens"].shape[1] == n)

    def summary(self):
        S = self.steps
        agg = dict(
            n_steps=len(S),
            n_pruning_decisions=int(sum(1 for _ in S)),
            all_decisions_match_oracle=all(s["decisions_match_oracle"] for s in S),
            all_decisions_match_fp64_oracle=all(s["decisions_match_fp64_oracle"] for s in S),
            max_score_abs_vs_oracle=max(s["score_max_abs_vs_oracle"] for s in S),
            max_score_abs_vs_fp64_oracle=max(s["score_max_abs_vs_fp64_oracle"] for s in S),
            max_logsoftmax_fp64_abs=max(s["logsoftmax_fp64_max_abs"] for s in S),
            max_sigma_abs=max(s["sigma_max_abs"] for s in S),
            max_cum_vs_fsum_abs=max(s["cum_vs_fsum_max_abs"] for s in S),
            counts_recount_exact=all(s["counts_recount_exact"] for s in S),
            prefix_length_ok=all(s["prefix_length_ok"] for s in S),
            sigma_nonneg=all(s["sigma_nonneg"] for s in S), logp_nonpos=all(s["logp_nonpos"] for s in S),
            prob_component_nonneg=all(s["prob_component_nonneg"] for s in S),
            propagation_ok=all(s.get("tokens_propagated", True) and s.get("hist_prefix_propagated", True)
                               and s.get("counts_propagated", True) and s.get("token_logp_propagated", True)
                               and s.get("cum_propagated_max_abs", 0.0) <= 1e-12 for s in S),
            max_cum_propagation_abs=max(s.get("cum_propagated_max_abs", 0.0) for s in S),
            steps_with_ebss_change=sum(s["sentences_where_ebss_changed_po_set"] for s in S),
            min_cut_gap=min(s["min_cut_gap"] for s in S),
            final=self.final, mismatches=self.mism)
        agg["passed"] = bool(agg["all_decisions_match_oracle"] and agg["all_decisions_match_fp64_oracle"]
                             and agg["max_score_abs_vs_oracle"] <= TOL_SCORE
                             and agg["max_logsoftmax_fp64_abs"] <= TOL_LSM and agg["max_sigma_abs"] <= 1e-12
                             and agg["max_cum_vs_fsum_abs"] <= TOL_SCORE and agg["counts_recount_exact"]
                             and agg["prefix_length_ok"] and agg["sigma_nonneg"] and agg["logp_nonpos"]
                             and agg["prob_component_nonneg"] and agg["propagation_ok"]
                             and self.final is not None and self.final["pick_matches"]
                             and self.final["final_score_max_abs"] <= TOL_SCORE and self.final["counts_exact"])
        return agg


def synthetic_trace(seed, ub, N=3, T=12, w=4, V=40, L=3, E=8, k=2):
    a = TraceAuditor(E)
    G.generate(G.SyntheticBackend(V=V, L=L, E=E, k=k, seed=seed), list(range(N)), seqlen=T, w=w, tau=TAU,
               use_balance=ub, audit=a)
    s = a.summary(); s["examples"] = a.examples[:3]
    return s


# =================================================================== stages
def stage_synthetic(a):
    t0 = time.time()
    ex = fixed_examples()
    jdump(dict(tolerances=dict(identity=TOL_ALG, production_vs_oracle=TOL_SCORE), examples=ex,
               tradeoff_threshold=tradeoff(), deferred_sigma_end_to_end=deferred_sigma_end_to_end(),
               passed=all(e["passed"] for e in ex)), p("synthetic_counterexamples.json"))
    jdump(cross_parent_trap(), p("cross_parent_sign_test.json"))
    jdump(metamorphic(), p("metamorphic_tests.json"))
    bw = beam_width_ranking()
    syn = {f"seed{s}_{'EBSS' if ub else 'PO'}": synthetic_trace(s, ub) for s in (0, 1, 2) for ub in (True, False)}
    # an extra configuration: large sigma influence (small tau) so that EBSS changes decisions often
    a2 = TraceAuditor(8)
    G.generate(G.SyntheticBackend(V=30, L=3, E=8, k=2, seed=7, logit_scale=0.5), [0, 1, 2], seqlen=16, w=4,
               tau=0.05, use_balance=True, audit=a2)
    syn["seed7_EBSS_tau0.05_flat_logits"] = dict(a2.summary(), examples=a2.examples[:3],
                                                 note="audit-only configuration (not a generation setting)")
    bw["synthetic_generate_traces"] = {k: {kk: v[kk] for kk in ("n_steps", "all_decisions_match_oracle",
                                                                "all_decisions_match_fp64_oracle",
                                                                "max_score_abs_vs_oracle", "propagation_ok",
                                                                "steps_with_ebss_change", "final", "passed")}
                                       for k, v in syn.items()}
    jdump(bw, p("beam_ranking_comparison.json"))
    jdump(syn, p("logs", "synthetic_generate_traces.json"))
    print(f"[synthetic] fixed {all(e['passed'] for e in ex)} trap {cross_parent_trap()['passed']} "
          f"beam {bw['passed']} traces {all(v['passed'] for v in syn.values())} ({time.time() - t0:.0f} s)")


def stage_historical(a):
    out = {}
    nll = torch.load(os.path.join(PILOT, "logs", "capture_routing_nll.pt"), weights_only=False)["nll"].numpy()
    man = json.load(open(os.path.join(PILOT, "logs", "capture_manifest.json")))["rows"]
    for name, f, key in [("EBSS", "generated_ebss.pt", "ebss"), ("PROBABILITY_ONLY", "generated_po.pt", "po")]:
        d = torch.load(os.path.join(PILOT, "logs", f), weights_only=False)
        ub = name == "EBSS"
        tau = d["config"]["tau"]; T = d["tokens"].shape[1]
        r = dict(n=int(d["tokens"].shape[0]), length=T)
        r["cum_vs_fsum_max_abs"] = float(max(abs(float(d["cum_logp"][i]) - O.cumulative_logp(d["token_logp"][i].numpy()))
                                             for i in range(r["n"])))
        oc = [O.counts_from_routing(d["hist_experts"][i].numpy(), 60) for i in range(r["n"])]
        r["counts_recount_exact"] = all(np.array_equal(oc[i], d["counts"][i].numpy()) for i in range(r["n"]))
        osig = [O.sigma_eq4(c, T) for c in oc]
        r["sigma_max_abs"] = float(max(abs(a_ - float(b)) for a_, b in zip(osig, d["sigma"])))
        fs = [O.final_score(O.cumulative_logp(d["token_logp"][i].numpy()), T, osig[i], tau, ub) for i in range(r["n"])]
        r["final_score_max_abs"] = float(max(abs(a_ - float(b)) for a_, b in zip(fs, d["final_score"])))
        fb = d["final_beams"]
        picks = []
        for i in range(r["n"]):
            fbs = [O.sigma_eq4(fb["counts"][i, b].numpy(), T)
                   for b in range(fb["cum_logp"].shape[1])]
            picks.append(O.final_select(fb["cum_logp"][i].numpy(), fbs, T, tau, ub))
        r["final_pick_matches_oracle"] = picks == fb["pick"].tolist()
        r["picked_beam_equals_output"] = all(torch.equal(fb["tokens"][i, int(fb["pick"][i])], d["tokens"][i])
                                             for i in range(r["n"]))
        r["logp_nonpos"] = bool((d["token_logp"] <= 0).all())
        # recorded top-8 step logs
        bad, n_c, gaps = [], 0, []
        for s in d["step_log"]:
            cs = s["candidates"]
            for c in cs:
                n_c += 1
                o = O.score_logp(c["R_parent"], c["logp_v"], s["prefix_len"], c["sigma_parent"], tau, ub)
                po_term = -(c["R_parent"] + c["logp_v"]) / (s["prefix_len"] + 1)
                if abs(o - c["score"]) > TOL_SCORE or abs(po_term - c["prob_term"]) > TOL_SCORE:
                    bad.append(dict(step=s["step"], sentence=s["sentence"], rank=c["rank"], oracle=o, logged=c["score"]))
            keys = [(c["score"], c["parent"], c["token"]) for c in cs]
            if keys != sorted(keys) or [c["kept"] for c in cs] != [c["rank"] < s["w"] for c in cs]:
                bad.append(dict(step=s["step"], sentence=s["sentence"], issue="order/kept flags"))
            if s["n_branches"] > 1 and ub:
                # deferred sigma: logged sigma_parent equals that parent's branch sigma
                for c in cs:
                    if abs(c["sigma_parent"] - s["branch_sigma"][c["parent"]]) > 0 or \
                            abs(c["R_parent"] - s["branch_R"][c["parent"]]) > 0:
                        bad.append(dict(step=s["step"], issue="parent state lookup"))
            gaps.append(cs[s["w"]]["score"] - cs[s["w"] - 1]["score"])
        r["step_log"] = dict(records=len(d["step_log"]), candidates_checked=n_c, mismatches=bad[:20],
                             n_mismatches=len(bad), min_cut_gap=float(min(gaps)),
                             limitation="only the top-8 of w*V candidates were logged at steps 1/2/64/256/511: "
                                         "ordering, scores and kept flags are verifiable; completeness of the "
                                         "top-w over the full pool is verified by the production capture stage")
        # B1 replay (teacher-forced, full 151936-row vocabulary, batch 1, no KV cache)
        rows = man[key]
        rep = nll[rows]
        gen = -d["cum_logp"].numpy() / (T - 1)
        r["teacher_forced_b1"] = dict(replay_mean_nll=rep.tolist(), generation_mean_nll=gen.tolist(),
                                      max_abs=float(np.abs(rep - gen).max()), mean_abs=float(np.abs(rep - gen).mean()),
                                      mean_signed=float((rep - gen).mean()),
                                      note="replay normalises over all 151936 logit rows, generation over 0..151645")
        r["passed"] = bool(r["cum_vs_fsum_max_abs"] <= TOL_SCORE and r["counts_recount_exact"]
                           and r["sigma_max_abs"] <= 1e-12 and r["final_score_max_abs"] <= TOL_SCORE
                           and r["final_pick_matches_oracle"] and r["picked_beam_equals_output"]
                           and r["logp_nonpos"] and not bad)
        out[name] = r
    jdump(out, p("logs", "historical_pilot_audit.json"))
    print("[historical]", {k: v["passed"] for k, v in out.items()})


def stage_production(a):
    sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
    import ebss_generation_pilot as EGP                     # noqa: E402  (same model loader as the pilot)
    t0 = time.time()
    be, load = EGP.load_backend(check_dispatch=True)
    starts = G.sample_first_tokens(4)                      # first 4 of the pilot's frozen start-token draw
    res = {}
    for ub in (True, False):
        aud = TraceAuditor(60, keep_examples=10)
        r = G.generate(be, starts, seqlen=32, w=W, tau=TAU, use_balance=ub, audit=aud)
        s = aud.summary()
        # teacher-forced replay of the chosen sentences (batch 1, no cache, same vocabulary slice)
        tf = []
        for i in range(r.tokens.shape[0]):
            lp, _ = be.replay(r.tokens[i].tolist())
            dlt = (lp - r.token_logp[i]).numpy()
            tf.append(dict(sentence=i, max_abs=float(np.abs(dlt).max()), mean_abs=float(np.abs(dlt).mean()),
                           cum_replay=float(lp.sum()), cum_generation=float(r.cum_logp[i]),
                           rel_cum=float(abs(lp.sum() - r.cum_logp[i]) / abs(r.cum_logp[i])),
                           pearson=float(np.corrcoef(lp.numpy(), r.token_logp[i].numpy())[0, 1])))
        allg = np.concatenate([[x["mean_abs"]] for x in tf])
        s["teacher_forced"] = dict(per_sentence=tf, mean_abs=float(allg.mean()),
                                   min_pearson=min(x["pearson"] for x in tf),
                                   flag=bool(allg.mean() > 0.05 or min(x["pearson"] for x in tf) < 0.99))
        # probability mass of the excluded embedding-padding rows (151646..151935) on the replayed sentences
        with torch.no_grad():
            ids = r.tokens.to(be.dev)
            full = be.model(input_ids=ids, use_cache=False).logits.float()
            lsm = torch.log_softmax(full, -1)
            excl = lsm[..., 151646:].exp().sum(-1)
        s["excluded_vocab_mass"] = dict(max=float(excl.max()), mean=float(excl.mean()))
        s["dispatch_mismatches"] = int(be.dispatch_mismatches)
        s["examples"] = aud.examples
        s["chosen_tokens"] = r.tokens.tolist()
        res["EBSS" if ub else "PROBABILITY_ONLY"] = s
        print(f"[production] {'EBSS' if ub else 'PO'} passed {s['passed']} decisions "
              f"{s['all_decisions_match_oracle']} ({time.time() - t0:.0f} s)", flush=True)
    res["load"] = load
    res["config"] = dict(n=4, seqlen=32, w=W, tau=TAU, starts=starts, vocab_limit=be.vocab_limit)
    jdump(res, p("production_trace_audit.json"))


def stage_report(a):
    syn = json.load(open(p("synthetic_counterexamples.json")))
    trap = json.load(open(p("cross_parent_sign_test.json")))
    meta = json.load(open(p("metamorphic_tests.json")))
    bw = json.load(open(p("beam_ranking_comparison.json")))
    hist = json.load(open(p("logs", "historical_pilot_audit.json")))
    prod = json.load(open(p("production_trace_audit.json"))) if os.path.exists(p("production_trace_audit.json")) \
        else None
    checks = dict(fixed_counterexamples=syn["passed"],
                  tradeoff_threshold=syn["tradeoff_threshold"]["passed"],
                  deferred_sigma_end_to_end=syn["deferred_sigma_end_to_end"]["passed"],
                  cross_parent_trap=trap["passed"], metamorphic=meta["passed"], beam_width_ranking=bw["passed"],
                  synthetic_generate_traces=all(v["passed"] for v in bw["synthetic_generate_traces"].values()),
                  historical_pilot=all(v["passed"] for v in hist.values()),
                  production_capture=prod is not None and all(prod[m]["passed"] for m in ("EBSS", "PROBABILITY_ONLY")))
    power = meta["wrong_variant_detection"]["sets_where_variant_changes_kept_set"]
    distinguishes = (trap["formulas"]["wrong_minus_nll_plus_logp"]["prefers"] == "B" and
                     all(v > 0 for v in power.values()))
    traceable = prod is not None
    if not all(checks.values()) and traceable and distinguishes:
        label = "SCORE-SIGN-BUG-CONFIRMED"
    elif not (traceable and distinguishes) or not all(checks.values()):
        label = "AUDIT-INCONCLUSIVE"
    else:
        label = "SCORE-SIGN-CORRECT"
    dec = dict(decision=label, secondary_finding="SCORE-CORRECT-PAPER-AMBIGUOUS" if label == "SCORE-SIGN-CORRECT"
               else None, checks=checks, wrong_variants_detectable=distinguishes, wrong_variant_power=power,
               rule=json.load(open(p("audit_protocol.json")))["decision_rule"],
               pilot_validity="B=16 pilot generation remains valid under the documented interpretation"
               if label == "SCORE-SIGN-CORRECT" else "requires revalidation")
    jdump(dec, p("decision.json"))
    print("[report]", label, checks)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["synthetic", "historical", "production", "report"])
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    dict(synthetic=stage_synthetic, historical=stage_historical, production=stage_production,
         report=stage_report)[a.stage](a)


if __name__ == "__main__":
    main()
