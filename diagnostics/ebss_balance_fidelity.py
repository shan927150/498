"""MoEQuant EBSS balance-strength fidelity experiment (Eq. 4 vs Algorithm 1 normalisation x rate vs share usage).

Stages (--stage):
  regression        CPU: independent normalisation oracle, default (EQ4_RATE) bitwise regression vs the frozen
                    pre-study generator and the stored pilot artifacts, invariance of everything except sigma
  short_generate    GPU: 4 x 128, PO + 4 variants, one backend load, pool capture (+ real-model default regression)
  counterfactual    CPU: oracle re-ranking of every captured candidate pool under all variants and PO
  short_diagnostics GPU: CANONICAL-B1 replay of the short sets (+ C4R_s0 truncated to 128) and all statistics
  gate              CPU: frozen scientific gate -> B=16 variant selection
  full_generate     GPU: --variant V, 16 x 512 (only if selected by the gate)
  full_diagnostics  GPU: CANONICAL-B1 capture of the new B=16 sets (+ C4R_s0 rows for cache compatibility), statistics
  quantize          GPU: --variant V, unchanged OV.quant_config on the new capture
  report            CPU: decision, summaries, figures
Nothing under runs_ebss_generation_pilot / runs_ebss_score_audit is written.
"""
from __future__ import annotations

import argparse, importlib.util, json, math, os, sys, time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "diagnostics"))
OUT = os.path.join(ROOT, "runs_ebss_balance_fidelity")
NCACHE = os.path.expanduser("~/moe_cache_ebss_balance_fidelity")
PILOT = os.path.join(ROOT, "runs_ebss_generation_pilot")
os.environ["EBSS_VALIDATION_CACHE"] = NCACHE
import official_ebss_validation as OV                  # noqa: E402  (ECACHE -> our cache: imported first)
OV.ECACHE = NCACHE
import ebss_generation_pilot as EGP                    # noqa: E402
OV.ECACHE = NCACHE
from moe_calib import ebss_generator as G              # noqa: E402
from moe_calib import ebss_pilot as P                  # noqa: E402

VARIANTS = list(G.BALANCE_VARIANTS)                    # EQ4_RATE, EQ4_SHARE, ALG1_SHARE, ALG1_RATE
ALT = [v for v in VARIANTS if v != "EQ4_RATE"]
COND = ["PROBABILITY_ONLY"] + VARIANTS
W, TAU, VOCAB, NL, E, K = G.W_REF, G.TAU_REF, 151646, 24, 60, 4
SHORT_N, SHORT_LEN = 4, 128
BINS = [(1, 32), (33, 64), (65, 128), (129, 256), (257, 511)]


def p(*a):
    return os.path.join(OUT, *a)


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    os.replace(tmp, path)


def jload(path):
    return json.load(open(path))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m                       # dataclasses need the module registered
    spec.loader.exec_module(m)
    return m


NO = _load(os.path.join(OUT, "normalization_oracle.py"), "normalization_oracle")
PROT = jload(p("protocol.json"))


def starts(n):
    s = jload(os.path.join(PILOT, "initialization_manifest.json"))["pilot_start_ids"][:n]
    if n == SHORT_N:
        assert s == PROT["short_generation"]["start_ids"]
    return s


def gen_kwargs(cond):
    return dict(use_balance=False) if cond == "PROBABILITY_ONLY" else dict(use_balance=True, **G.BALANCE_VARIANTS[cond])


# =================================================================== pool capture (read-only audit hook)
class PoolRecorder:
    """Keeps, per step and row, the top-(w+1) tokens by (logP desc, token asc) of every parent, the parent's cum
    logP and routing counts, and the production decision. Because sigma is constant within a parent, the global
    top-(w+1) under ANY per-parent sigma lies inside this reduced pool."""

    def __init__(self, w, keep_state_at=()):
        self.w, self.keep_at = w, set(keep_state_at)
        self.steps, self.state, self.final = [], {}, None
        self.hook_s = 0.0

    def __call__(self, r):
        t0 = time.time()
        if r["kind"] == "final":
            self.final = dict(final_scores=r["final_scores"].numpy(), pick=r["pick"].numpy())
            self.hook_s += time.time() - t0
            return
        lp = r["logp"]
        R = lp.shape[0]
        m = self.w + 1
        toks = torch.empty(R, m, dtype=torch.long); vals = torch.empty(R, m, dtype=torch.float64)
        for i in range(R):
            thr = torch.topk(lp[i], m).values[-1]
            cand = torch.nonzero(lp[i] >= thr).view(-1)                       # ascending token ids
            o = torch.sort(-lp[i, cand], stable=True).indices[:m]
            toks[i] = cand[o]; vals[i] = lp[i, cand[o]].double()
        self.steps.append(dict(step=r["step"], l=r["prefix_len"], B=r["n_branches"], cum=r["cum"].numpy().copy(),
                               cand_tok=toks.numpy(), cand_logp=vals.numpy(),
                               counts=r["counts"].numpy().astype(np.int16), sigma_prod=r["sigma"].numpy().copy(),
                               parent=r["parent"].numpy().copy(), token=r["token"].numpy().copy()))
        if r["step"] in self.keep_at:
            self.state[r["step"]] = dict(tokens=r["tokens"].clone(), cum=r["cum"].clone(), sigma=r["sigma"].clone())
        self.hook_s += time.time() - t0


# =================================================================== regression (CPU)
def oracle_tests():
    out = []

    def rec(name, ok, **kw):
        out.append(dict(test=name, passed=bool(ok), **kw))
    rng = np.random.default_rng(20261110)
    # explicit counts: 2 layers x 6 experts, n = 6 tokens, k = 2
    C = np.array([[6, 2, 2, 1, 1, 0], [3, 3, 2, 2, 1, 1]])
    n, k = 6, 2
    for v in VARIANTS:
        s, u = G.BALANCE_VARIANTS[v]["balance_std_mode"], G.BALANCE_VARIANTS[v]["usage_mode"]
        o = NO.sigma(C.tolist(), n, s, u, k)
        pr = float(G.balance_sigma(torch.tensor(C), n, s, u, k))
        # hand computation for layer 0: u_rate = [1, 1/3, 1/3, 1/6, 1/6, 0], mean 1/3
        rec(f"production_equals_oracle[{v}]", abs(o - pr) <= 1e-15, oracle=o, production=pr)
    ur = np.array([1, 1 / 3, 1 / 3, 1 / 6, 1 / 6, 0]); ss = float(((ur - ur.mean()) ** 2).sum())
    hand = dict(eq4_rate=math.sqrt(ss / 5), alg1_rate=math.sqrt(6 * ss / 5), eq4_share=math.sqrt(ss / 5) / 2,
                alg1_share=math.sqrt(6 * ss / 5) / 2)
    for key, (s, u) in dict(eq4_rate=("eq4", "per_token"), alg1_rate=("algorithm1", "per_token"),
                            eq4_share=("eq4", "assignment_share"), alg1_share=("algorithm1", "assignment_share")).items():
        o = NO.layer_sigma(C[0].tolist(), n, s, u, k)
        rec(f"hand_layer0[{key}]", abs(o - hand[key]) <= 1e-15, hand=hand[key], oracle=o)
    # analytic ratios at E = 60, k = 4 and E = 8, k = 2 (no hard-coded 60)
    for E_, k_ in ((60, 4), (8, 2)):
        for _ in range(20):
            Ct = rng.integers(0, 50, size=(3, E_)); nt = int(rng.integers(5, 400))
            base = float(G.balance_sigma(torch.tensor(Ct), nt, top_k=k_))
            for v in VARIANTS:
                vv = G.BALANCE_VARIANTS[v]
                val = float(G.balance_sigma(torch.tensor(Ct), nt, vv["balance_std_mode"], vv["usage_mode"], k_))
                expect = NO.analytic_ratio(v, E_, k_)
                ok = abs(val / base - expect) <= 1e-12 * expect if base > 0 else val == 0
                if not ok:
                    rec(f"analytic_ratio[{v},E={E_}]", False, ratio=val / base, expect=expect)
        rec(f"analytic_ratios_E{E_}_k{k_}", True, ratios={v: NO.analytic_ratio(v, E_, k_) for v in VARIANTS})
    rec("sqrtE_relation_E60", abs(NO.analytic_ratio("ALG1_RATE", 60) - math.sqrt(60)) < 1e-15, value=math.sqrt(60))
    rec("share_is_rate_over_k", abs(NO.analytic_ratio("EQ4_SHARE", 60) - 0.25) < 1e-15)
    rec("strengths_table", True, strengths={v: NO.analytic_ratio(v, 60) for v in VARIANTS})
    # uniform -> 0 ; nonuniform -> > 0 ; heterogeneous expert counts per layer (oracle)
    for v in VARIANTS:
        vv = G.BALANCE_VARIANTS[v]
        uni = float(G.balance_sigma(torch.full((24, 60), 8), 120, vv["balance_std_mode"], vv["usage_mode"], 4))
        non = float(G.balance_sigma(torch.tensor(C), n, vv["balance_std_mode"], vv["usage_mode"], 2))
        rec(f"uniform_zero_nonuniform_positive[{v}]", uni == 0.0 and non > 0, uniform=uni, nonuniform=non)
    het = NO.sigma([[4, 0, 0, 0], [2, 2, 1, 1, 0, 0, 0, 2]], 2, "algorithm1", "per_token", 2)
    het_hand = (math.sqrt(4 / 3 * sum((x - 0.5) ** 2 for x in [2, 0, 0, 0])) +
                math.sqrt(8 / 7 * sum((x - 0.5) ** 2 for x in [1, 1, .5, .5, 0, 0, 0, 1]))) / 2
    rec("heterogeneous_E_per_layer_oracle", abs(het - het_hand) < 1e-15, oracle=het, hand=het_hand,
        note="production stores counts as one [L, E] tensor; the Qwen checkpoint has E = 60 at all 24 layers")
    # more balanced parent with identical likelihood scores lower; deferred prefix; cross-parent
    import ebss_score_audit as A
    for v in VARIANTS:
        vv = G.BALANCE_VARIANTS[v]
        sb = float(G.balance_sigma(torch.tensor([[2, 2, 2, 2]]), 2, vv["balance_std_mode"], vv["usage_mode"], 4))
        si = float(G.balance_sigma(torch.tensor([[8, 0, 0, 0]]), 2, vv["balance_std_mode"], vv["usage_mode"], 4))
        sc = G.candidate_scores(torch.tensor([-1.0, -1.0], dtype=torch.float64),
                                torch.tensor([[-0.5], [-0.5]], dtype=torch.float64),
                                torch.tensor([sb, si], dtype=torch.float64), 2, TAU, True)
        rec(f"balanced_parent_scores_lower[{v}]", sc[0, 0] < sc[1, 0], balanced=float(sc[0, 0]), imbalanced=float(sc[1, 0]))
        ok_, info = deferred_variant(A, v)
        rec(f"deferred_prefix_cross_parent_generate[{v}]", ok_, **info)
    return out


def deferred_variant(A, v):
    """Score-audit scripted test with the variant's normalisation: EBSS keeps both children of the balanced parent;
    parent sigmas equal the oracle's variant sigma of the PARENT prefix."""
    vv = G.BALANCE_VARIANTS[v]

    def logits_fn(pf):
        return [math.log(.1), math.log(.4), math.log(.4), math.log(.05), math.log(.05)] if len(pf) == 1 \
            else [math.log(.2)] * 5

    def route_fn(pf):
        return [[2, 3]] if pf[-1] == 2 else [[0, 1]]
    rec = []
    G.generate(A.ScriptedBackend(5, 1, 4, 2, logits_fn, route_fn), [4], seqlen=3, w=2, tau=TAU, use_balance=True,
               audit=rec.append, **vv)
    st = [x for x in rec if x["kind"] == "step" and x["step"] == 2][0]
    hist = st["hist"].numpy()
    osig = [NO.sigma_variant(A.O.counts_from_routing(hist[b], 4).tolist(), 2, v, k=2) for b in range(2)]
    kept_parent_tok = [int(st["tokens"][int(q)][1]) for q in st["parent"]]
    ok = (max(abs(a - b) for a, b in zip(osig, st["sigma"].tolist())) <= 1e-15 and kept_parent_tok == [2, 2])
    return ok, dict(parent_sigma_oracle=osig, parent_sigma_production=st["sigma"].tolist(),
                    kept_children_of_parent_token=kept_parent_tok)


def default_regression_synthetic():
    old = _load(p("logs", "ebss_generator_pre_fidelity.py"), "ebss_generator_pre_fidelity")
    res = []
    for seed in (0, 1, 2, 3):
        for ub in (True, False):
            for explicit in (False, True):
                ro, rn = [], []
                kw = dict(seqlen=14, w=4, tau=TAU, use_balance=ub, record_steps=(1, 2, 7))
                a = old.generate(old.SyntheticBackend(V=40, L=3, E=8, k=2, seed=seed), [0, 1, 2], audit=ro.append, **kw)
                b = G.generate(G.SyntheticBackend(V=40, L=3, E=8, k=2, seed=seed), [0, 1, 2], audit=rn.append,
                               **(dict(kw, **G.BALANCE_DEFAULT) if explicit else kw))
                same = all(torch.equal(getattr(a, f), getattr(b, f)) for f in
                           ("tokens", "cum_logp", "token_logp", "counts", "hist_experts", "sigma", "final_score"))
                same &= all(torch.equal(a.final_beams[f], b.final_beams[f]) for f in a.final_beams)
                same &= a.step_log == b.step_log and a.intervention == b.intervention
                steps = all(all(torch.equal(x[f], y[f]) for f in ("scores", "sigma", "parent", "token", "cum", "counts"))
                            for x, y in zip(ro[:-1], rn[:-1])) and len(ro) == len(rn)
                res.append(dict(seed=seed, use_balance=ub, options_explicit=explicit, outputs_bitwise=bool(same),
                                every_step_scores_and_decisions_bitwise=bool(steps)))
    return dict(cases=res, passed=all(r["outputs_bitwise"] and r["every_step_scores_and_decisions_bitwise"] for r in res))


def default_regression_historical():
    """Stored pilot counts: new default vs frozen pre-study function on the same device (bitwise), and vs the values
    stored by the pilot (computed on the GPU: equality up to device reduction order is reported, tolerance 1e-15)."""
    old = _load(p("logs", "ebss_generator_pre_fidelity.py"), "ebss_generator_pre_fidelity")
    out = {}
    for m in ("ebss", "po"):
        d = torch.load(os.path.join(PILOT, "logs", f"generated_{m}.pt"), weights_only=False)
        fbc, cum = d["final_beams"]["counts"], d["final_beams"]["cum_logp"].double()
        s, s0 = G.balance_sigma(d["counts"], 512), old.balance_sigma(d["counts"], 512)
        fb, fb0 = G.balance_sigma(fbc, 512), old.balance_sigma(fbc, 512)
        fs = G.final_scores(cum, fb, 512, TAU, m == "ebss")
        fs0 = old.final_scores(cum, fb0, 512, TAU, m == "ebss")
        out[m] = dict(sigma_bitwise_vs_frozen=bool(torch.equal(s, s0)),
                      final_beam_sigma_bitwise_vs_frozen=bool(torch.equal(fb, fb0)),
                      final_scores_bitwise_vs_frozen=bool(torch.equal(fs, fs0)),
                      sigma_vs_stored_max_abs=float((s - d["sigma"]).abs().max()),
                      final_scores_vs_stored_max_abs=float((fs - d["final_beams"]["score"]).abs().max()),
                      final_pick_reproduced=bool(torch.equal(torch.argmin(fs, 1), d["final_beams"]["pick"])))
        out[m]["passed"] = bool(out[m]["sigma_bitwise_vs_frozen"] and out[m]["final_beam_sigma_bitwise_vs_frozen"]
                                and out[m]["final_scores_bitwise_vs_frozen"] and out[m]["sigma_vs_stored_max_abs"] <= 1e-15
                                and out[m]["final_scores_vs_stored_max_abs"] <= 1e-15 and out[m]["final_pick_reproduced"])
    out["passed"] = all(out[m]["passed"] for m in ("ebss", "po"))
    return out


def invariance_tests():
    """All variants share everything except sigma: PO identical under every variant's options; on one frozen pool
    the EBSS scores minus the PO scores equal sigma_v / tau exactly per parent."""
    res = {}
    po_ref = G.generate(G.SyntheticBackend(seed=11), [0, 1], seqlen=12, w=4, tau=TAU, use_balance=False)
    for v in VARIANTS:
        r = G.generate(G.SyntheticBackend(seed=11), [0, 1], seqlen=12, w=4, tau=TAU, use_balance=False,
                       **G.BALANCE_VARIANTS[v])
        rec = []
        G.generate(G.SyntheticBackend(seed=11), [0, 1], seqlen=8, w=4, tau=TAU, use_balance=True, audit=rec.append,
                   **G.BALANCE_VARIANTS[v])
        st = [x for x in rec if x["kind"] == "step" and x["n_branches"] > 1][2]
        po = G.candidate_scores(st["cum"], st["logp"], st["sigma"], st["prefix_len"], TAU, False)
        diff = st["scores"].reshape(po.shape) - po
        dec = (diff - (st["sigma"] / TAU)[:, None]).abs().max().item()     # (s + x) - s vs x: rounding only
        same_po = all(torch.equal(getattr(po_ref, f), getattr(r, f)) for f in ("tokens", "cum_logp", "token_logp",
                                                                                "counts", "hist_experts"))
        cfg = r.config
        res[v] = dict(po_bitwise_identical_under_variant_options=bool(same_po),
                      score_minus_po_equals_sigma_over_tau_max_abs=dec,
                      config_serialised=dict(balance_std_mode=cfg["balance_std_mode"], usage_mode=cfg["usage_mode"],
                                             top_k=cfg["top_k"]),
                      passed=bool(same_po and dec <= 1e-13 and cfg["balance_std_mode"] ==
                                  G.BALANCE_VARIANTS[v]["balance_std_mode"]))
    res["passed"] = all(v["passed"] for k, v in res.items() if k != "passed")
    return res


def stage_regression(a):
    ot = oracle_tests()
    jdump(dict(tests=ot, passed=all(t["passed"] for t in ot)), p("normalization_oracle_results.json"))
    dr = dict(synthetic_vs_frozen_pre_study_generator=default_regression_synthetic(),
              historical_pilot_artifacts=default_regression_historical(), invariance=invariance_tests(),
              real_model="filled by stage short_generate")
    dr["passed_cpu"] = bool(dr["synthetic_vs_frozen_pre_study_generator"]["passed"]
                            and dr["historical_pilot_artifacts"]["passed"] and dr["invariance"]["passed"])
    jdump(dr, p("default_regression.json"))
    print(f"[regression] oracle {all(t['passed'] for t in ot)} default/invariance {dr['passed_cpu']}")
    if not (dr["passed_cpu"] and all(t["passed"] for t in ot)):
        raise SystemExit("regression FAILED")


# =================================================================== generation helpers (GPU)
def run_generation(be, cond, n, seqlen, record_steps=(), keep_state_at=(), progress_every=0):
    torch.cuda.reset_peak_memory_stats()
    kv_peak = [0]
    orig = be.step

    def step(tokens):
        out = orig(tokens)
        kv_peak[0] = max(kv_peak[0], be.kv_cache_bytes())
        return out
    be.step = step
    n0 = (be.n_forward, be.rows_forwarded, be.bytes_transferred)
    pr = PoolRecorder(W, keep_state_at)
    t0 = time.time()
    try:
        r = G.generate(be, starts(n), seqlen, W, TAU, record_steps=record_steps, audit=pr,
                       progress_every=progress_every, **gen_kwargs(cond))
    finally:
        be.step = orig
    wall = time.time() - t0
    nf, rows, by = be.n_forward - n0[0], be.rows_forwarded - n0[1], be.bytes_transferred - n0[2]
    acc = P.token_accounting(n, seqlen, W, VOCAB)
    t = r.timing
    cost = dict(condition=cond, wall_s=wall, gpu_seconds=wall, wall_excl_pool_capture_s=wall - pr.hook_s,
                pool_capture_hook_s=pr.hook_s, timing=t, forward_calls=nf, rows_forwarded=rows,
                avg_forward_batch=rows / nf, bytes_host_to_device=by,
                h2d_gb_per_s=by / 1e9 / max(t.get("expert_transfer_s", 1e-9), 1e-9), accounting=acc,
                emitted_tokens_per_s=acc["emitted_tokens"] / wall, scored_candidates_per_s=acc["scored_candidates"] / wall,
                peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9, kv_cache_peak_gb=kv_peak[0] / 1e9,
                peak_cpu_gb=EGP.FD.peak_cpu_gb(), validation=P.validate_generated(r.tokens, n, seqlen, VOCAB),
                gpu_seconds_note="one MIG slice used exclusively; GPU-seconds = wall (includes the read-only pool capture)")
    compute = t["forward_s"] - t["expert_transfer_s"] - t["routing_hook_s"]
    parts = dict(model_forward_compute=compute, expert_weight_transfer=t["expert_transfer_s"],
                 routing_capture_hook=t["routing_hook_s"], expert_statistics_update=t["stats_s"],
                 beam_scoring=t["score_s"], beam_pruning_selection=t["select_s"],
                 beam_state_gather_and_kv_reorder=t["reorder_gather_s"], pool_capture_audit=pr.hook_s)
    parts["other"] = t["total_s"] - sum(parts.values())
    cost["breakdown_s"] = parts
    cost["breakdown_frac"] = {k: v / t["total_s"] for k, v in parts.items()}
    gen = dict(tokens=r.tokens, cum_logp=r.cum_logp, token_logp=r.token_logp, counts=r.counts,
               hist_experts=r.hist_experts, sigma=r.sigma, final_score=r.final_score, final_beams=r.final_beams,
               step_log=r.step_log, intervention=r.intervention, config=r.config, method=cond)
    return gen, cost, pr


def save_pools(pr, path):
    torch.save(dict(steps=pr.steps, final=pr.final), path)


def stage_short_generate(a):
    dr = jload(p("default_regression.json"))
    assert dr["passed_cpu"] and jload(p("normalization_oracle_results.json"))["passed"], "CPU regression not passed"
    out_pt = p("logs", "short_generations.pt")
    if os.path.exists(out_pt):
        print("[short] done"); return
    be, load = EGP.load_backend(check_dispatch=False)
    gens, costs = {}, {}
    audit = jload(os.path.join(ROOT, "runs_ebss_score_audit", "production_trace_audit.json"))
    real = {}
    for cond in COND:
        g, c, pr = run_generation(be, cond, SHORT_N, SHORT_LEN, keep_state_at=(32,))
        gens[cond], costs[cond] = g, c
        save_pools(pr, p("logs", f"pools_short_{cond}.pt"))
        if cond in ("PROBABILITY_ONLY", "EQ4_RATE"):
            st = pr.state[32]
            ub = cond == "EQ4_RATE"
            fs = G.final_scores(st["cum"], st["sigma"], 32, TAU, ub).view(SHORT_N, W)
            pick = torch.argmin(fs, 1)
            toks = st["tokens"].view(SHORT_N, W, 32)[torch.arange(SHORT_N), pick]
            ref = audit["EBSS" if ub else "PROBABILITY_ONLY"]["chosen_tokens"]
            real[cond] = dict(length32_selection_equals_score_audit_outputs=toks.tolist() == ref)
        print(f"[short] {cond}: {c['wall_s']:.0f} s valid {c['validation']['valid']}", flush=True)
    real["passed"] = all(v["length32_selection_equals_score_audit_outputs"] for v in real.values())
    dr["real_model"] = dict(real, note="beam state of the 4 x 128 run at |S| = 32 put through the length-32 final "
                                       "selection vs the score audit's 4 x 32 production outputs (same starts/layout)")
    dr["passed"] = bool(dr["passed_cpu"] and real["passed"])
    jdump(dr, p("default_regression.json"))
    torch.save(gens, out_pt)
    jdump(dict(load=load, costs=costs), p("logs", "short_generation_cost.json"))
    print(f"[short] real-model default regression {real}")
    if not dr["passed"]:
        raise SystemExit("real-model default regression FAILED")


# =================================================================== counterfactual (CPU)
def counterfactual_pools(path):
    d = torch.load(path, weights_only=False)
    rows = []
    for s in d["steps"]:
        B, l = s["B"], s["l"]
        if B == 1:
            continue
        R = s["cum"].shape[0]
        Nsent = R // B
        for i in range(Nsent):
            sl = slice(i * B, (i + 1) * B)
            par = np.repeat(np.arange(B), W + 1)
            tok = s["cand_tok"][sl].reshape(-1)
            lp = s["cand_logp"][sl]
            cum = s["cum"][sl]
            sig = {v: np.array([NO.sigma_variant(s["counts"][sl][b].astype(np.int64).tolist(), l, v, K)
                                for b in range(B)]) for v in VARIANTS}
            base = NO.scores(cum, lp, np.zeros(B), l, TAU, False)          # probability component [B, w+1]
            res = {}
            for c in COND:
                sc = base if c == "PROBABILITY_ONLY" else base + (sig[c] / TAU)[:, None]
                keep, order = NO.rank(sc.reshape(-1), par, tok, W)
                flat = sc.reshape(-1)
                res[c] = dict(kept=sorted((int(par[j]), int(tok[j])) for j in keep),
                              min_score=float(flat[order[0]]), margin=float(flat[order[W]] - flat[order[W - 1]]),
                              n_parents=len({int(par[j]) for j in keep}))
            prod = sorted(zip((s["parent"][i * W:(i + 1) * W] - i * B).tolist(), s["token"][i * W:(i + 1) * W].tolist()))
            best_child = base[:, 0]
            rows.append(dict(step=s["step"], l=l, sentence=i, prod=prod, res=res,
                             prob_spread=float(best_child.max() - best_child.min()),
                             bal_spread={v: float((sig[v] / TAU).max() - (sig[v] / TAU).min()) for v in VARIANTS},
                             sigma_eq4_rate_prod_abs=float(np.abs(sig["EQ4_RATE"] - s["sigma_prod"][sl]).max())))
    return rows


def summarise_cf(rows, traj):
    def frac(sel, a, b):
        x = [r for r in rows if sel(r)]
        return (sum(r["res"][a]["kept"] != r["res"][b]["kept"] for r in x) / len(x)) if x else None, len(x)
    out = dict(trajectory=traj, n_sentence_steps=len(rows))
    if traj in COND:                              # oracle reproduces the production decision of its own trajectory
        out["oracle_reproduces_production_decisions"] = all(r["res"][traj]["kept"] == r["prod"] for r in rows)
        if traj != "PROBABILITY_ONLY":
            out["max_abs_sigma_oracle_vs_production"] = max(r["sigma_eq4_rate_prod_abs"] for r in rows) \
                if traj == "EQ4_RATE" else None
    per = {}
    for v in COND:
        dv, n = frac(lambda r: True, v, "EQ4_RATE")
        dpo, _ = frac(lambda r: True, v, "PROBABILITY_ONLY")
        bins = {}
        for lo, hi in BINS:
            x, m = frac(lambda r: lo <= r["l"] <= hi, v, "EQ4_RATE")
            y, _ = frac(lambda r: lo <= r["l"] <= hi, v, "PROBABILITY_ONLY")
            bins[f"{lo}-{hi}"] = dict(n=m, D_vs_EQ4_RATE=x, D_vs_PO=y)
        rat = [r["bal_spread"][v] / r["prob_spread"] for r in rows if v in r["bal_spread"] and r["prob_spread"] > 0]
        per[v] = dict(D_vs_EQ4_RATE=dv, n_changed_vs_EQ4_RATE=int(round(dv * n)) if dv is not None else None,
                      D_vs_PO=dpo, n_changed_vs_PO=int(round(dpo * n)) if dpo is not None else None, by_prefix_bin=bins,
                      mean_parents_in_kept=float(np.mean([r["res"][v]["n_parents"] for r in rows])),
                      median_margin=float(np.median([r["res"][v]["margin"] for r in rows])),
                      median_min_score=float(np.median([r["res"][v]["min_score"] for r in rows])),
                      median_balance_spread=float(np.median([r["bal_spread"][v] for r in rows])) if v in VARIANTS else 0.0,
                      median_balance_to_probability_spread=float(np.median(rat)) if rat else 0.0)
    out["median_probability_spread"] = float(np.median([r["prob_spread"] for r in rows]))
    out["per_condition"] = per
    return out


def stage_counterfactual(a):
    res, raw = {}, {}
    for cond in COND:
        f = p("logs", f"pools_short_{cond}.pt")
        rows = counterfactual_pools(f)
        res[f"short_{cond}"] = summarise_cf(rows, cond)
        raw[f"short_{cond}"] = rows
    for v in ALT:
        f = p("logs", f"pools_full_{v}.pt")
        if os.path.exists(f):
            rows = counterfactual_pools(f)
            res[f"full_{v}"] = summarise_cf(rows, v)
            raw[f"full_{v}"] = rows
    torch.save(raw, p("logs", "counterfactual_rows.pt"))
    res["primary"] = "short_EQ4_RATE"
    res["caveat"] = PROT["counterfactual"]["caveat"]
    jdump(res, p("counterfactual_ranking.json"))
    pr = res["short_EQ4_RATE"]
    print("[counterfactual] primary oracle==production", pr.get("oracle_reproduces_production_decisions"),
          {v: (round(pr["per_condition"][v]["D_vs_EQ4_RATE"], 3), round(pr["per_condition"][v]["D_vs_PO"], 3))
           for v in COND})


# =================================================================== diagnostics helpers
def load_reference():
    from moe_calib import typicality as TY
    prof = np.load(os.path.join(ROOT, "runs_low_budget", "routing_profile_v2.npz"))
    ref = TY.pool_reference(prof["C"], 4, E)
    pool = TY.sequence_scores(prof["C"], 4, E, ref=ref)
    typ_ids = jload(os.path.join(ROOT, "runs_typicality", "pool_typical_half.json"))["ids"]
    return ref, float(np.nanmax(pool["d"][np.array(typ_ids)]))


def set_stats(sel_b1, tokens, nll, gen_hist, ref, d_star, tokzr):
    """Identical diagnostic implementation for every condition (pilot blocks)."""
    import diagnostics_ebss_pilot as DG
    out = dict(b1=DG.exposure_block(DG.counts_from_sel(sel_b1)),
               representativeness=DG.representativeness(sel_b1, ref, d_star),
               text=DG.text_block(tokens, nll, tokzr))
    for blk in ("b1",):
        out[blk].pop("histogram", None)
    if gen_hist is not None:
        g = DG.exposure_block(DG.counts_from_sel(gen_hist)); g.pop("histogram", None)
        out["generation_time"] = g
        agree = np.sort(gen_hist, -1) == np.sort(sel_b1, -1)
        out["generation_vs_b1_set_agreement"] = float(agree.all(-1).mean())
    return out


def compact(st):
    b, g = st["b1"]["mean_over_layers"], st.get("generation_time", {}).get("mean_over_layers", {})
    t, rp = st["text"], st["representativeness"]
    return dict(paper_sigma_canonical=st["b1"]["paper_sigma"], cv=b["cv"], gini=b["gini"], min=b["min"], p10=b["p10"],
                n_zero=b["n_zero"], per_layer_cv=[x["cv"] for x in st["b1"]["per_layer"]],
                gen_cv=g.get("cv"), gen_min=g.get("min"), gen_p10=g.get("p10"),
                gen_paper_sigma=st.get("generation_time", {}).get("paper_sigma"),
                model_ppl=t["model_nll"]["corpus_ppl"], distinct1=t["distinct1_mean"], distinct2=t["distinct2_mean"],
                distinct3=t["distinct3_mean"], repetition_rate=t["repetition_rate_mean"],
                longest_identical_run_max=t["longest_identical_token_run"]["max"],
                longest_identical_run_median=t["longest_identical_token_run"]["median"],
                longest_repeated_ngram_mean=t["longest_repeated_ngram"]["mean"],
                near_duplicate_pairs_ge_0_5=t["near_duplicates"].get("pairs_jaccard_ge_0.5"),
                zlib_ratio_median=t["zlib_ratio"]["median"], routing_l1=rp["routing_l1_mean"],
                frac_atypical=rp["frac_atypical"], gen_vs_b1_agreement=st.get("generation_vs_b1_set_agreement"))


def stage_short_diagnostics(a):
    from transformers import AutoTokenizer
    gens = torch.load(p("logs", "short_generations.pt"), weights_only=False)
    for c in COND:
        assert gens[c]["tokens"].shape == (SHORT_N, SHORT_LEN) and P.validate_generated(
            gens[c]["tokens"], SHORT_N, SHORT_LEN, VOCAB)["valid"]
    man = jload(os.path.join(PILOT, "logs", "capture_manifest.json"))
    pid = torch.load(os.path.join(P.CACHE, "ids.pt"))
    pos = {r: i for i, r in enumerate(man["rows"]["c4_rows"])}
    c4_idx = [pos[r] for r in man["c4_random"]["0"]["chosen"]]
    c4_tok = pid["tokens"][c4_idx][:, :SHORT_LEN]
    tokens = torch.cat([gens[c]["tokens"] for c in COND] + [c4_tok])
    model, t_load = EGP.FD.load_model_full()
    assert OV.verify_pristine_full(model)
    t0 = time.time()
    _, sel, nll = EGP.forward_b1_sel(model, tokens)
    t_cap = time.time() - t0
    sel = sel.numpy()
    psel = torch.load(os.path.join(PILOT, "logs", "capture_routing_nll.pt"))["sel"].numpy()[c4_idx][:, :SHORT_LEN]
    c4_trunc_agree = float((np.sort(sel[-16:], -1) == np.sort(psel, -1)).all(-1).mean())
    ref, d_star = load_reference()
    tokzr = AutoTokenizer.from_pretrained(P.MODEL)
    res = {}
    for j, c in enumerate(COND):
        rr = slice(j * SHORT_N, (j + 1) * SHORT_N)
        st = set_stats(sel[rr], gens[c]["tokens"].tolist(), nll[rr], gens[c]["hist_experts"].numpy(), ref, d_star, tokzr)
        st["internal_sigma_mean"] = float(gens[c]["sigma"].mean()) if c != "PROBABILITY_ONLY" else None
        st["intervention"] = gens[c]["intervention"]
        st["final_tokens_identical_to_EQ4_RATE"] = [bool(torch.equal(gens[c]["tokens"][i], gens["EQ4_RATE"]["tokens"][i]))
                                                    for i in range(SHORT_N)]
        st["first_divergence_vs_EQ4_RATE"] = first_div(gens[c]["tokens"], gens["EQ4_RATE"]["tokens"])
        st["first_divergence_vs_PO"] = first_div(gens[c]["tokens"], gens["PROBABILITY_ONLY"]["tokens"])
        res[c] = st
    c4 = slice(len(COND) * SHORT_N, None)
    res["C4R_s0_first128"] = set_stats(sel[c4], c4_tok.tolist(), nll[c4], None, ref, d_star, tokzr)
    res["C4R_s0_first128_first4"] = set_stats(sel[c4][:4], c4_tok[:4].tolist(), nll[c4][:4], None, ref, d_star, tokzr)
    comp = {c: compact(v) for c, v in res.items()}
    beam = beam_divergence()
    jdump(dict(config=PROT["short_generation"], compact=comp, beam_behaviour=beam, full=res,
               capture_s=t_cap, model_load_s=t_load,
               c4_first128_vs_pilot_512_capture_set_agreement=c4_trunc_agree,
               note="all balance outcomes use the canonical EQ4_RATE definition; internal_sigma_mean is each "
                    "variant's own score term and is not comparable across variants"),
          p("short_generation_results.json"))
    print("[short_diag]", {c: (round(comp[c]["cv"], 4), round(comp[c]["distinct2"], 3)) for c in comp})


def first_div(a, b):
    out = []
    for i in range(a.shape[0]):
        d = (a[i] != b[i]).nonzero()
        out.append(int(d[0]) if len(d) else None)
    return out


def beam_divergence():
    """Trajectory-level: first step at which a condition's retained beam set differs from EQ4_RATE's / PO's, and the
    fraction of steps with different retained sets (after divergence the prefixes differ, so the fraction is
    descriptive only)."""
    kept = {}
    for c in COND:
        d = torch.load(p("logs", f"pools_short_{c}.pt"), weights_only=False)["steps"]
        kept[c] = [[tuple(sorted(zip(s["parent"][i * W:(i + 1) * W].tolist(), s["token"][i * W:(i + 1) * W].tolist())))
                    for i in range(SHORT_N)] for s in d]
    out = {}
    for c in COND:
        row = {}
        for ref in ("EQ4_RATE", "PROBABILITY_ONLY"):
            first, frac = [], []
            for i in range(SHORT_N):
                diff = [kept[c][j][i] != kept[ref][j][i] for j in range(len(kept[c]))]
                first.append(next((j + 1 for j, x in enumerate(diff) if x), None))
                frac.append(float(np.mean(diff)))
            row[f"vs_{ref}"] = dict(first_divergent_step=first, frac_steps_retained_set_differs=frac)
        out[c] = row
    return out


# =================================================================== gate
def rel(a, b):
    return (a - b) / abs(b) if b else float("inf")


def stage_gate(a):
    cf = jload(p("counterfactual_ranking.json"))["short_EQ4_RATE"]["per_condition"]
    sh = jload(p("short_generation_results.json"))["compact"]
    ref, po = sh["EQ4_RATE"], sh["PROBABILITY_ONLY"]
    stats = ("cv", "p10", "min", "distinct2", "repetition_rate", "longest_repeated_ngram_mean", "routing_l1",
             "frac_atypical")
    res = {}
    for v in ALT:
        s, c = sh[v], cf[v]
        g = dict(
            G1_beam=dict(D_v=c["D_vs_EQ4_RATE"], n_changed=c["n_changed_vs_EQ4_RATE"],
                         pass_=c["D_vs_EQ4_RATE"] >= 0.05 and c["n_changed_vs_EQ4_RATE"] >= 5),
            G2_balance=dict(rel_cv=rel(s["cv"], ref["cv"]), rel_cv_gen=rel(s["gen_cv"], ref["gen_cv"]),
                            pass_=abs(rel(s["cv"], ref["cv"])) >= 0.10
                            and np.sign(rel(s["cv"], ref["cv"])) == np.sign(rel(s["gen_cv"], ref["gen_cv"]))),
            G3_lower_tail=dict(rel_p10=rel(s["p10"], ref["p10"]), rel_min=rel(s["min"], ref["min"]),
                               rel_p10_gen=rel(s["gen_p10"], ref["gen_p10"]), rel_min_gen=rel(s["gen_min"], ref["gen_min"]),
                               pass_=bool((abs(rel(s["p10"], ref["p10"])) >= 0.10 and np.sign(rel(s["p10"], ref["p10"]))
                                           == np.sign(rel(s["gen_p10"], ref["gen_p10"])))
                                          or (abs(rel(s["min"], ref["min"])) >= 0.20 and np.sign(rel(s["min"], ref["min"]))
                                              == np.sign(rel(s["gen_min"], ref["gen_min"]))))),
            G4_repetition=dict(d_distinct2=s["distinct2"] - ref["distinct2"],
                               d_repetition=s["repetition_rate"] - ref["repetition_rate"],
                               rel_longest_ngram=rel(s["longest_repeated_ngram_mean"], ref["longest_repeated_ngram_mean"]),
                               pass_=bool(abs(s["distinct2"] - ref["distinct2"]) >= 0.05
                                          or abs(s["repetition_rate"] - ref["repetition_rate"]) >= 0.05
                                          or abs(rel(s["longest_repeated_ngram_mean"],
                                                     ref["longest_repeated_ngram_mean"])) >= 0.25)),
            G5_routing=dict(rel_l1=rel(s["routing_l1"], ref["routing_l1"]),
                            d_atypical=s["frac_atypical"] - ref["frac_atypical"],
                            pass_=bool(abs(rel(s["routing_l1"], ref["routing_l1"])) >= 0.10
                                       or abs(s["frac_atypical"] - ref["frac_atypical"]) >= 0.25)))
        for kk in g:
            g[kk]["pass_"] = bool(g[kk]["pass_"])
        g["qualifies"] = any(x["pass_"] for x in g.values())
        res[v] = g
    context = {k: dict(PO_minus_EQ4_RATE=po[k] - ref[k], rel=rel(po[k], ref[k])) for k in stats}
    alg1 = [v for v in ("ALG1_SHARE", "ALG1_RATE") if res[v]["qualifies"]]
    sel = list(alg1)
    if len(sel) < 2 and res["EQ4_SHARE"]["qualifies"]:
        sel.append("EQ4_SHARE")
    excluded = {v: ("did not qualify (no gate criterion passed)" if not res[v]["qualifies"] else
                    "qualified but lower fidelity priority than the two Algorithm-1 variants (max two new variants)")
                for v in ALT if v not in sel}
    material = any(res[v]["qualifies"] for v in ALT)
    out = dict(thresholds=PROT["scientific_gate"], per_variant=res, po_vs_eq4_rate_context=context,
               selected_for_B16=sel, excluded=excluded, short_stage_material=material)
    jdump(out, p("scientific_gate.json"))
    print("[gate]", {v: res[v]["qualifies"] for v in ALT}, "-> B16:", sel)


# =================================================================== full generation / diagnostics / quantize
def stage_full_generate(a):
    v = a.variant
    gate = jload(p("scientific_gate.json"))
    assert v in gate["selected_for_B16"], f"{v} not selected by the frozen gate"
    pt = p("logs", f"generated_{v}.pt")
    if os.path.exists(pt):
        print(f"[full] {v} done"); return
    be, load = EGP.load_backend(check_dispatch=False)
    g, c, pr = run_generation(be, v, 16, 512, record_steps=EGP.PILOT_RECORD_STEPS, progress_every=64)
    c["load"] = load
    if not c["validation"]["valid"]:
        raise SystemExit(f"{v} generated set invalid")
    torch.save(g, pt)
    save_pools(pr, p("logs", f"pools_full_{v}.pt"))
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(P.MODEL)

    class _R:
        pass
    r = _R(); r.method = v
    for kk in ("tokens", "cum_logp", "sigma", "final_score"):
        setattr(r, kk, g[kk])
    P.write_jsonl(r, p(f"generated_{v}.jsonl"), t.decode)
    jdump(c, p("logs", f"generation_cost_{v}.json"))
    print(f"[full] {v}: {c['wall_s']:.0f} s valid {c['validation']['valid']}")


def stage_full_diagnostics(a):
    from transformers import AutoTokenizer
    gate = jload(p("scientific_gate.json"))
    new = [v for v in gate["selected_for_B16"] if os.path.exists(p("logs", f"generated_{v}.pt"))]
    assert new, "no B=16 generation"
    man_path = p("logs", "capture_manifest.json")
    pman = jload(os.path.join(PILOT, "logs", "capture_manifest.json"))
    pid = torch.load(os.path.join(P.CACHE, "ids.pt"))
    pos = {r: i for i, r in enumerate(pman["rows"]["c4_rows"])}
    c4_idx = [pos[r] for r in pman["c4_random"]["0"]["chosen"]]
    gens = {v: torch.load(p("logs", f"generated_{v}.pt"), weights_only=False) for v in new}
    if not os.path.exists(man_path):
        tokens = torch.cat([pid["tokens"][c4_idx]] + [gens[v]["tokens"] for v in new])
        rows = dict(c4r_s0=list(range(16)), **{v: list(range(16 * (j + 1), 16 * (j + 2))) for j, v in enumerate(new)})
        model, t_load = EGP.FD.load_model_full()
        assert OV.verify_pristine_full(model)
        t0 = time.time()
        stored, sel, nll = EGP.forward_b1_sel(model, tokens)
        t_cap = time.time() - t0
        os.makedirs(NCACHE, exist_ok=True)
        files, reg = {}, {}
        for L in EGP.LAYERS:
            path = os.path.join(NCACHE, f"L{L}.pt")
            torch.save(stored[L], path + ".tmp"); os.replace(path + ".tmp", path)
            files[str(L)] = dict(shape=list(stored[L].shape), sha256=EGP.sha256_file(path))
            R = torch.load(os.path.join(P.CACHE, f"L{L}.pt"), mmap=True)
            reg[str(L)] = bool(torch.equal(stored[L][:16], R[c4_idx]))
        json.dump(dict(pool_size=0, holdout_size=0, seqlen=512, layers=EGP.LAYERS, policy="CANONICAL-B1",
                       n_rows=len(tokens)), open(os.path.join(NCACHE, "meta.json"), "w"), indent=1)
        torch.save(dict(tokens=tokens, rows=rows), os.path.join(NCACHE, "ids.pt"))
        torch.save(dict(sel=sel, nll=torch.tensor(nll)), p("logs", "capture_routing_nll.pt"))
        psel = torch.load(os.path.join(PILOT, "logs", "capture_routing_nll.pt"))["sel"][c4_idx]
        man = dict(cache_dir=NCACHE, rows=rows, files=files, capture_s=t_cap, model_load_s=t_load,
                   compatibility=dict(c4r_s0_rows_bitwise_equal_pilot_cache=reg,
                                      c4r_s0_routing_equal_pilot=bool(torch.equal(sel[:16], psel))),
                   peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
        man["passed"] = all(reg.values()) and man["compatibility"]["c4r_s0_routing_equal_pilot"]
        jdump(man, man_path)
        print(f"[full_diag] capture {t_cap:.0f} s compatibility {man['passed']}")
    man = jload(man_path)
    cap = torch.load(p("logs", "capture_routing_nll.pt"))
    psel_all = torch.load(os.path.join(PILOT, "logs", "capture_routing_nll.pt"))
    ref, d_star = load_reference()
    tokzr = AutoTokenizer.from_pretrained(P.MODEL)
    pgens = {m: torch.load(os.path.join(PILOT, "logs", f"generated_{m}.pt"), weights_only=False) for m in ("po", "ebss")}
    res = {}
    for name, m in (("PROBABILITY_ONLY", "po"), ("EQ4_RATE", "ebss")):
        rr = pman["rows"][m]
        res[name] = set_stats(psel_all["sel"].numpy()[rr], pid["tokens"][rr].tolist(), psel_all["nll"].numpy()[rr],
                              pgens[m]["hist_experts"].numpy(), ref, d_star, tokzr)
        res[name]["intervention"] = pgens[m]["intervention"]
    for s in ("0", "1", "2"):
        rr = [pos[r] for r in pman["c4_random"][s]["chosen"]]
        res[f"C4R_s{s}"] = set_stats(psel_all["sel"].numpy()[rr], pid["tokens"][rr].tolist(),
                                     psel_all["nll"].numpy()[rr], None, ref, d_star, tokzr)
    for v in new:
        rr = man["rows"][v]
        res[v] = set_stats(cap["sel"].numpy()[rr], gens[v]["tokens"].tolist(), cap["nll"].numpy()[rr],
                           gens[v]["hist_experts"].numpy(), ref, d_star, tokzr)
        res[v]["intervention"] = gens[v]["intervention"]
        res[v]["internal_sigma_mean"] = float(gens[v]["sigma"].mean())
        res[v]["first_divergence_vs_EQ4_RATE"] = first_div(gens[v]["tokens"], pgens["ebss"]["tokens"])
        res[v]["identical_sequences_vs_EQ4_RATE"] = int(sum(torch.equal(gens[v]["tokens"][i], pgens["ebss"]["tokens"][i])
                                                            for i in range(16)))
    # contamination of the new sets against held-out evaluation data
    allt = OV.c4_tokens()
    win, _ = EGP.EP.load_windows()
    held = [allt[i].tolist() for i in range(2000, 2256)] + [w.tolist() for w in win]
    cont = {v: P.contamination(gens[v]["tokens"].tolist(), held) for v in new}
    comp = {c: compact(x) for c, x in res.items()}
    ebc = comp["EQ4_RATE"]
    per_layer = {v: int(sum(a < b for a, b in zip(comp[v]["per_layer_cv"], comp["PROBABILITY_ONLY"]["per_layer_cv"])))
                 for v in comp if v not in ("PROBABILITY_ONLY",) and not v.startswith("C4R")}
    jdump(dict(compact=comp, full=res, contamination=cont, capture_manifest=man,
               layers_cv_improved_vs_PO=per_layer,
               gate_thresholds_at_B16={v: dict(rel_cv=rel(comp[v]["cv"], ebc["cv"]), rel_p10=rel(comp[v]["p10"], ebc["p10"]),
                                               rel_min=rel(comp[v]["min"], ebc["min"]),
                                               d_distinct2=comp[v]["distinct2"] - ebc["distinct2"],
                                               d_repetition=comp[v]["repetition_rate"] - ebc["repetition_rate"],
                                               rel_l1=rel(comp[v]["routing_l1"], ebc["routing_l1"]),
                                               d_atypical=comp[v]["frac_atypical"] - ebc["frac_atypical"]) for v in new}),
          p("full_generation_results.json"))
    print("[full_diag]", {c: (round(comp[c]["cv"], 4), round(comp[c]["distinct2"], 3)) for c in comp})


def stage_quantize(a):
    v = a.variant
    os.makedirs(p("results"), exist_ok=True)
    out = p("results", f"{v}.json")
    if os.path.exists(out):
        print(f"[quant] {v} done"); return
    man = jload(p("logs", "capture_manifest.json"))
    assert man["passed"], "capture not compatible with the pilot cache"
    for L in EGP.LAYERS:
        assert EGP.sha256_file(os.path.join(NCACHE, f"L{L}.pt")) == man["files"][str(L)]["sha256"], "cache changed"
    chosen = man["rows"][v]
    assert len(chosen) == 16 == len(set(chosen))
    model, t_load = EGP.FD.load_model_full()
    assert OV.verify_pristine_full(model)
    assert OV.ECACHE == NCACHE
    r = OV.quant_config(model, v, NCACHE, chosen, OV.OCACHE, 4)
    r.update(config=v, policy="CANONICAL-B1", cache_rows=chosen, model_load_s=t_load, B=16)
    jdump(r, out)
    print(f"[quant] {v}: recon {r['recon']:.7f} PPL {r['ppl']:.6f} RTN fallbacks {r['rtn_fallbacks']} valid {r['valid']}",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--variant", choices=ALT)
    a = ap.parse_args(argv)
    os.makedirs(p("logs"), exist_ok=True)
    if a.stage == "report":
        import report_ebss_balance_fidelity as R
        return R.main()
    dict(regression=stage_regression, short_generate=stage_short_generate, counterfactual=stage_counterfactual,
         short_diagnostics=stage_short_diagnostics, gate=stage_gate, full_generate=stage_full_generate,
         full_diagnostics=stage_full_diagnostics, quantize=stage_quantize)[a.stage](a)


if __name__ == "__main__":
    main()
