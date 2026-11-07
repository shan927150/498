"""Locked independent confirmation split (final aggregate experiment).

Stage --replay (CPU, tokenizer only)
  Replays moe_calib.data_utils.get_splits EXACTLY (same C4 shard
  en/c4-train.00000-of-01024, data_seed 0, same shuffle / skip / random-window
  rule, seqlen 512) with need = 2256 + 256, recording for every window its
  document index and start offset. The loop is the same code path, so the
  first 2256 windows are a prefix of the replay; they must reproduce
  ~/moe_cache/ids.pt (pool + holdout) token for token, or the stage aborts.
  Confirmation = the first 256 windows AFTER the first 2256, in replay order,
  that pass all checks below against every earlier window (pool, tune, test
  and already-accepted confirmation windows); rejected windows are recorded.
  (Rule fixed after the first replay found one near-duplicate - a templated
  SEO page sharing ~200 tokens with pool row 1261 - and before any
  evaluation.) Checks:
    * document indices disjoint from all 2256 earlier windows (one window per
      document by construction - asserted, not assumed);
    * no identical document TEXT (sha256 of the raw text) shared with any
      earlier document (duplicate C4 documents under different indices);
    * no shared 32-token shingle with any earlier window (near-duplicates);
    * length 512, same tokenizer.
  Writes <out>/ids.pt {"confirmation": [256, 512]} and <out>/provenance.json
  (doc indices, starts, text hashes, fingerprint). Never touches ~/moe_cache.

Stage --cache (GPU, full model)
  Embeds and runs the full model layer by layer exactly like cache_inputs.py
  (batch 8) and saves the INPUT hidden states of layers 1/8/16/23 to
  <out>/L{L}.pt ([256, 512, 2048] bf16). As a pipeline check it recomputes the
  batch-aligned old test rows 2128..2135 the same way and compares them with
  ~/moe_cache (recorded in provenance.json).
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SHARD = "en/c4-train.00000-of-01024.json.gz"
N_OLD, N_NEW, SEQLEN, DATA_SEED, SHINGLE = 2256, 256, 512, 0, 32
N_EXTRA = 64            # replay headroom for rejected near-duplicates


def replay(tokenizer, need, seqlen=SEQLEN, data_seed=DATA_SEED):
    """get_splits' loop verbatim, plus provenance."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", data_files={"train": SHARD}, split="train")
    rng = random.Random(data_seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    out, prov = [], []
    for idx in order:
        if len(out) >= need:
            break
        text = ds[idx]["text"]
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        if ids.shape[0] <= seqlen:
            continue
        st = rng.randint(0, ids.shape[0] - seqlen - 1)
        out.append(ids[st:st + seqlen])
        prov.append(dict(doc=int(idx), start=int(st), n_tokens=int(ids.shape[0]),
                         text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                         url=ds[idx].get("url")))
    return out, prov


def fingerprint(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def shingles(t, k=SHINGLE):
    t = t.tolist()
    return {hash(tuple(t[i:i + k])) for i in range(len(t) - k + 1)}


def stage_replay(out_dir, cache_dir):
    from transformers import AutoTokenizer
    from moe_calib.model_utils import MODEL_NAME
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    t0 = time.time()
    wins, prov = replay(tok, N_OLD + N_NEW + N_EXTRA)
    assert len(wins) == N_OLD + N_NEW + N_EXTRA
    old = torch.load(os.path.join(cache_dir, "ids.pt"))
    old_ids = torch.cat([old["pool"], old["holdout"]])
    rep = torch.stack(wins[:N_OLD])
    if old_ids.shape != rep.shape or not torch.equal(old_ids, rep):
        raise SystemExit("replay does NOT reproduce the existing pool/tune/test ids: abort")
    old_docs = {p["doc"] for p in prov[:N_OLD]}
    assert len(old_docs) == N_OLD, "one window per document violated in the existing splits"
    seen_docs, seen_txt, seen_sh = set(old_docs), {p["text_sha256"] for p in prov[:N_OLD]}, set()
    for w in rep:
        seen_sh |= shingles(w)
    acc, rejected = [], []
    for j in range(N_OLD, len(wins)):
        if len(acc) >= N_NEW:
            break
        p, w = prov[j], wins[j]
        sh = shingles(w)
        why = [r for r, bad in (("doc_index", p["doc"] in seen_docs), ("doc_text", p["text_sha256"] in seen_txt),
                                ("shingle32", bool(sh & seen_sh)), ("length", w.shape[0] != SEQLEN)) if bad]
        if why:
            rejected.append(dict(replay_pos=j, doc=p["doc"], url=p["url"], reasons=why))
            continue
        acc.append(j)
        seen_docs.add(p["doc"]); seen_txt.add(p["text_sha256"]); seen_sh |= sh
    if len(acc) < N_NEW:
        raise SystemExit(f"only {len(acc)} clean windows in the replay headroom; increase N_EXTRA")
    new = torch.stack([wins[j] for j in acc])
    assert new.shape == (N_NEW, SEQLEN)
    newp = [dict(prov[j], replay_pos=j) for j in acc]
    new_docs = [p["doc"] for p in newp]
    old_sh = set()
    for w in rep:
        old_sh |= shingles(w)
    doc_overlap = sorted(old_docs & set(new_docs))
    old_txt = {p["text_sha256"] for p in prov[:N_OLD]}
    txt_overlap = [p["doc"] for p in newp if p["text_sha256"] in old_txt]
    new_txt_dup = N_NEW - len({p["text_sha256"] for p in newp})
    sh_overlap = [i for i, w in enumerate(new) if shingles(w) & old_sh]
    exact = [i for i, w in enumerate(new) if (rep == w).all(1).any()]
    roles = {"pool": (0, 2000), "tune": (2000, 2128), "test": (2128, 2256)}
    res = dict(shard=SHARD, data_seed=DATA_SEED, seqlen=SEQLEN, tokenizer=MODEL_NAME,
               replay_need=len(wins), reproduces_existing_ids=True,
               existing_ids_fingerprint=fingerprint(old_ids),
               selection_rule="first 256 replay windows after position 2256 sharing no document index, "
                              "document text or 32-token shingle with any earlier accepted window",
               rejected_windows=rejected,
               confirmation_size=N_NEW, confirmation_fingerprint=fingerprint(new),
               document_index_overlap=doc_overlap, document_text_overlap=txt_overlap,
               duplicate_texts_within_confirmation=new_txt_dup,
               windows_sharing_a_32token_shingle_with_old=sh_overlap,
               exact_window_matches=exact, roles_checked=roles,
               previously_used_for_selection_or_tuning=False,
               replay_seconds=time.time() - t0,
               confirmation=newp, old_role_docs={
                   r: [p["doc"] for p in prov[a:b]] for r, (a, b) in roles.items()})
    ok = not doc_overlap and not txt_overlap and not exact and not new_txt_dup and not sh_overlap
    res["disjoint_ok"] = ok
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "ids.pt")):
        prev = torch.load(os.path.join(out_dir, "ids.pt"))["confirmation"]
        if not torch.equal(prev, new):
            if os.path.exists(os.path.join(out_dir, "L1.pt")):
                raise SystemExit("an existing cached confirmation split differs from the replay: refusing")
            print("replacing the pre-filter ids.pt (no activations were cached from it)")
            torch.save({"confirmation": new}, os.path.join(out_dir, "ids.pt"))
    else:
        torch.save({"confirmation": new}, os.path.join(out_dir, "ids.pt"))
    json.dump(res, open(os.path.join(out_dir, "provenance.json"), "w"), indent=1)
    print(f"replay ok; confirmation fingerprint {res['confirmation_fingerprint'][:16]}; doc overlap "
          f"{len(doc_overlap)}, text overlap {len(txt_overlap)}, shingle overlap {len(sh_overlap)}, "
          f"exact {len(exact)}; disjoint_ok={ok}")
    if not ok:
        raise SystemExit("confirmation split is not document-disjoint: investigate before proceeding")


def stage_cache(out_dir, cache_dir, layers=(1, 8, 16, 23), batch=8):
    from moe_calib import model_utils as MU
    from moe_calib.data_utils import load_layer_cache
    prov = json.load(open(os.path.join(out_dir, "provenance.json")))
    new = torch.load(os.path.join(out_dir, "ids.pt"))["confirmation"]
    assert fingerprint(new) == prov["confirmation_fingerprint"]
    old = torch.load(os.path.join(cache_dir, "ids.pt"))
    chk_rows = list(range(2128, 2136))                       # batch-aligned old test rows
    chk = old["holdout"][[r - 2000 for r in chk_rows]]
    model, _ = MU.load_model()
    t0 = time.time()
    d = MU.expert_dims(model)[0]
    hs = torch.empty(N_NEW, SEQLEN, d, dtype=MU.DTYPE)
    for i in range(0, N_NEW, batch):
        hs[i:i + batch] = MU.embed(model, new[i:i + batch]).cpu()
    hc = MU.embed(model, chk).cpu()
    last = max(layers)
    check = {}
    for L in range(last + 1):
        if L in layers:
            torch.save(hs.clone(), os.path.join(out_dir, f"L{L}.pt"))
            ref = load_layer_cache(cache_dir, L)[chk_rows]
            check[str(L)] = dict(bit_exact=bool(torch.equal(ref, hc)),
                                 max_abs_diff=float((ref.float() - hc.float()).abs().max()))
            print(f"[conf-cache] saved L{L}.pt; old-row check {check[str(L)]}", flush=True)
        if L == last:
            break
        with MU.layer_on_gpu(model, L):
            for i in range(0, N_NEW, batch):
                hs[i:i + batch] = MU.run_layer(model, L, hs[i:i + batch].to(MU.GPU)).cpu()
            hc = MU.run_layer(model, L, hc.to(MU.GPU)).cpu()
    meta = dict(kind="confirmation", n=N_NEW, seqlen=SEQLEN, layers=list(layers), model=MU.MODEL_NAME,
                fingerprint=prov["confirmation_fingerprint"], source_cache=cache_dir, batch=batch,
                cache_seconds=time.time() - t0, pipeline_check_old_test_rows=chk_rows, pipeline_check=check)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/moe_cache_confirmation"))
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/moe_cache"))
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--cache", action="store_true")
    a = ap.parse_args(argv)
    if a.replay:
        stage_replay(a.out, a.cache_dir)
    if a.cache:
        stage_cache(a.out, a.cache_dir)


if __name__ == "__main__":
    main()
