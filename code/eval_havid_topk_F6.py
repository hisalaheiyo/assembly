"""
Extend F6 (unseen top-K recovery) to HAViD = 3/3 datasets.

Computes HAViD action top-{1,3,5,10} for the two F6 curves using the SAME
methodology as MECCANO/IMPACT (eval_{impact,meccano}_indep_action.py):
  action score(v,m) = logP(verb=v) + logP(manip=m) over the VALID (verb,manip)
  composite vocabulary, ranked by topk_acc (the shared train_impact_mvitv2.topk_acc).

Same checkpoints as the locked HAViD results (no retraining):
  shared (k2)  -> checkpoints/havid_gran_k2_{sub}/best.pth          (T5/F4/S2)
  independent  -> checkpoints/havid_gran_solo_{verb,manip}_{sub}/best.pth
                  (the recombination comprehensive_eval.load_havid uses for T2)
  atomic (k1)  -> unseen=0 for all K by construction (no eval needed).

DATA-INTEGRITY SAFEGUARDS
  * read-only on every existing file; writes ONLY results/havid_topk_F6.json (new).
  * reproducibility GATE: in the same pass we also compute the UNRESTRICTED
    both-correct top-1 (argmax verb & argmax manip) and compare against the locked
    numbers (k2 joint_top1; comprehensive independent unseen/seen top1). If any
    subtask deviates by > TOL, the script ABORTS and writes nothing.
  * no checkpoint written; deterministic eval (is_train=False).

Usage:  python eval_havid_topk_F6.py            # all 6 subtasks, both configs
        python eval_havid_topk_F6.py --subtask view0_lh_aa   # one (debug)
"""
import argparse, json, sys, statistics as st
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_mvitv2_comp import HAViDDataset, MViTv2MultiHead, HEAD_CLASSES, DATA_BASE, PROJECT_DIR
from train_impact_mvitv2 import topk_acc

N_VERB, N_MANIP = HEAD_CLASSES[0], HEAD_CLASSES[1]          # 11, 42
SUB = [f"view{v}_{h}h_aa" for v in (0, 1, 2) for h in ("l", "r")]
A_K = [1, 3, 5, 10]
TOL = 1.0          # pp; reproducibility gate tolerance vs locked numbers
RES = PROJECT_DIR / "results"
CKPT = PROJECT_DIR / "checkpoints"


def read_pairs(ann):
    """(verb,manip) pairs from a HAViD comp annotation file (one-hot fields 6,7)."""
    out = []
    with open(ann) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            v = parts[6].index("1") if "1" in parts[6] else -1
            m = parts[7].index("1") if "1" in parts[7] else -1
            if v >= 0 and m >= 0:
                out.append((v, m))
    return out


def build_vocab(subtask):
    """Valid composite vocabulary = union of (v,m) over train + val_seen + val_unseen
    (the analogue of IMPACT's load_action_to_vn valid-action set)."""
    dr = DATA_BASE / subtask
    pairs = set()
    for split in ("comp_train", "comp_val_seen", "comp_val_unseen"):
        pairs.update(read_pairs(dr / f"{subtask}_{split}.txt"))
    vocab = sorted(pairs)                      # deterministic order
    pair2id = {p: i for i, p in enumerate(vocab)}
    av = torch.tensor([p[0] for p in vocab], dtype=torch.long)
    am = torch.tensor([p[1] for p in vocab], dtype=torch.long)
    return pair2id, av, am


@torch.no_grad()
def eval_pass(get_logits, loader, pair2id, av, am, device):
    """get_logits(clip) -> (verb_logits, manip_logits).
    Returns restricted action top-K (logP(v)+logP(m) over vocab) AND the
    unrestricted both-correct top-1 (gate)."""
    av = av.to(device); am = am.to(device)
    n = 0
    a_c = {k: 0 for k in A_K}
    both = 0
    for frames, labels4 in tqdm(loader, desc="eval", leave=False):
        frames = frames.to(device, non_blocking=True)
        gv = labels4[:, 0].to(device); gm = labels4[:, 1].to(device)
        v_lg, m_lg = get_logits(frames)
        bs = frames.size(0); n += bs
        # --- restricted action scoring over the valid composite vocab ---
        log_pv = F.log_softmax(v_lg, dim=1)
        log_pm = F.log_softmax(m_lg, dim=1)
        scores = log_pv[:, av] + log_pm[:, am]          # (B, |vocab|)
        gt_cid = torch.tensor([pair2id[(int(a), int(b))] for a, b in zip(gv.tolist(), gm.tolist())],
                              device=device)
        ca = topk_acc(scores, gt_cid, A_K)
        for k in A_K:
            a_c[k] += ca[k]
        # --- unrestricted both-correct top-1 (reproducibility gate) ---
        both += int(((v_lg.argmax(1) == gv) & (m_lg.argmax(1) == gm)).sum())
    return ({f"action_top{k}": a_c[k] / n * 100 for k in A_K},
            both / n * 100, n)


def loaders(subtask, bs, nw):
    dr = DATA_BASE / subtask
    vids = [dr / "train" / "videos_train", dr / "val" / "videos_val"]
    seen = HAViDDataset(dr / f"{subtask}_comp_val_seen.txt", vids, is_train=False)
    unseen = HAViDDataset(dr / f"{subtask}_comp_val_unseen.txt", vids, is_train=False)
    dl = lambda ds: DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                               pin_memory=True, persistent_workers=False)
    return dl(seen), dl(unseen)


def load_model(classes, ckpt_path, device):
    assert ckpt_path.exists(), f"MISSING ckpt: {ckpt_path}"
    m = MViTv2MultiHead(classes, dropout=0.0, pretrained=False).to(device)
    c = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(c["model"]); m.eval()
    return m, c.get("epoch", -1)


def locked_k2(subtask):
    d = json.load(open(RES / f"havid_gran_k2_{subtask}" / "comp_eval.json"))
    return d["seen"]["joint_top1"], d["unseen"]["joint_top1"]


def locked_indep(subtask):
    d = json.load(open(RES / f"comprehensive_havid_indep_{subtask}" / "summary.json"))
    # both-correct == d3 'both' fraction == aggregate top1
    return (d["d3_seen"]["fractions_pct"]["both"],
            d["d3_unseen"]["fractions_pct"]["both"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subtask", default=None, help="single subtask (debug); default = all 6")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--dry_run", action="store_true", help="eval + gate but do not write")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subs = [args.subtask] if args.subtask else SUB
    print(f"[device] {device} | subtasks: {subs}")

    per_sub = {}            # subtask -> {config -> {seen:{}, unseen:{}}}
    gate_fail = []
    for sub in subs:
        pair2id, av, am = build_vocab(sub)
        seen_ld, unseen_ld = loaders(sub, args.batch_size, args.num_workers)
        print(f"\n=== {sub} | |vocab|={len(pair2id)} ===")

        # ---------- shared (k2) ----------
        k2_ckpt = CKPT / f"havid_gran_k2_{sub}" / "best.pth"
        m, ep = load_model([N_VERB, N_MANIP], k2_ckpt, device)
        gl = lambda fr: (lambda o: (o[0], o[1]))(m(fr))
        s_topk, s_both, s_n = eval_pass(gl, seen_ld, pair2id, av, am, device)
        u_topk, u_both, u_n = eval_pass(gl, unseen_ld, pair2id, av, am, device)
        del m; torch.cuda.empty_cache() if device.type == "cuda" else None
        ls, lu = locked_k2(sub)
        ds, du = abs(s_both - ls), abs(u_both - lu)
        ok = ds <= TOL and du <= TOL
        print(f"  [shared k2] ckpt={k2_ckpt.name}(e{ep}) "
              f"GATE both seen {s_both:.2f} vs locked {ls:.2f} (Δ{ds:.2f}); "
              f"unseen {u_both:.2f} vs {lu:.2f} (Δ{du:.2f}) -> {'OK' if ok else 'FAIL'}")
        print(f"             restricted action top-K seen={ [round(s_topk[f'action_top{k}'],1) for k in A_K] } "
              f"unseen={ [round(u_topk[f'action_top{k}'],1) for k in A_K] }")
        if not ok: gate_fail.append((sub, "shared", ds, du))

        # ---------- independent (solo recombination) ----------
        vck = CKPT / f"havid_gran_solo_verb_{sub}" / "best.pth"
        mck = CKPT / f"havid_gran_solo_manip_{sub}" / "best.pth"
        vm, vep = load_model([N_VERB], vck, device)
        mm, mep = load_model([N_MANIP], mck, device)
        gl2 = lambda fr: (vm(fr)[0], mm(fr)[0])
        s2_topk, s2_both, _ = eval_pass(gl2, seen_ld, pair2id, av, am, device)
        u2_topk, u2_both, _ = eval_pass(gl2, unseen_ld, pair2id, av, am, device)
        del vm, mm; torch.cuda.empty_cache() if device.type == "cuda" else None
        lis, liu = locked_indep(sub)
        di_s, di_u = abs(s2_both - lis), abs(u2_both - liu)
        ok2 = di_s <= TOL and di_u <= TOL
        print(f"  [independent] verb(e{vep})+manip(e{mep}) "
              f"GATE both seen {s2_both:.2f} vs locked {lis:.2f} (Δ{di_s:.2f}); "
              f"unseen {u2_both:.2f} vs {liu:.2f} (Δ{di_u:.2f}) -> {'OK' if ok2 else 'FAIL'}")
        print(f"               restricted action top-K seen={ [round(s2_topk[f'action_top{k}'],1) for k in A_K] } "
              f"unseen={ [round(u2_topk[f'action_top{k}'],1) for k in A_K] }")
        if not ok2: gate_fail.append((sub, "independent", di_s, di_u))

        per_sub[sub] = {
            "shared":      {"seen": s_topk,  "unseen": u_topk,
                            "gate_both": {"seen": s_both, "unseen": u_both, "locked": [ls, lu]}},
            "independent": {"seen": s2_topk, "unseen": u2_topk,
                            "gate_both": {"seen": s2_both, "unseen": u2_both, "locked": [lis, liu]}},
        }

    # ---------- gate verdict ----------
    if gate_fail:
        print("\n[ABORT] reproducibility gate FAILED (nothing written):")
        for sub, cfg, a, b in gate_fail:
            print(f"   {sub} {cfg}: Δseen={a:.2f} Δunseen={b:.2f} (TOL={TOL})")
        sys.exit(1)

    # ---------- aggregate (mean over subtasks) ----------
    def agg(cfg, split):
        return {f"action_top{k}": st.mean(per_sub[s][cfg][split][f"action_top{k}"] for s in subs)
                for k in A_K}
    out = {"datasets": "HAViD", "metric": "action(verb,manip) top-K, logP(v)+logP(m) over valid composites",
           "n_subtasks": len(subs), "subtasks": subs, "TOL_pp": TOL,
           "aggregate": {cfg: {sp: agg(cfg, sp) for sp in ("seen", "unseen")}
                         for cfg in ("shared", "independent")},
           "per_subtask": per_sub}
    if args.dry_run or args.subtask:
        print("\n[dry/single] not writing the aggregate file.")
    else:
        out_path = RES / "havid_topk_F6.json"
        assert not out_path.exists() or True   # new file; safe to (over)write only THIS file
        json.dump(out, open(out_path, "w"), indent=2,
                  default=lambda o: float(o))
        print(f"\n[done] gate PASSED on all {len(subs)} subtasks -> {out_path}")
    # echo headline
    for cfg in ("shared", "independent"):
        u = out["aggregate"][cfg]["unseen"]
        print(f"  HAViD {cfg:11s} unseen top-K = {[round(u[f'action_top{k}'],1) for k in A_K]}")


if __name__ == "__main__":
    main()
