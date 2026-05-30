"""
IMPACT 'independent' architecture evaluation on the compositional split.

Combines the independently-trained verb_only + noun_only models (each its own
backbone) and recombines the action via log P(v) + log P(n), mirroring the
MECCANO independent recombination (eval_meccano_indep_action.py).

Reports seen / unseen / HM action top-K and the per-component error
decomposition (D3) for the independent architecture.

Usage:
  python eval_impact_indep_action.py --view front --hand left
"""

import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_impact_mvitv2 import (
    action_logits_from_vn, build_action_lookup, topk_acc, VIEWS, HANDS,
)
from train_impact_comp import IMPACTCompDataset, PROJECT_DIR, hm
from train_impact_comp_variants import SingleHeadModel, N_VERB, N_NOUN


@torch.no_grad()
def evaluate(verb_model, noun_model, loader, device, split):
    verb_model.eval(); noun_model.eval()
    n = 0
    a_K = [1, 3, 5, 10]
    a_c = {k: 0 for k in a_K}
    cell = {"both": 0, "verb_only": 0, "noun_only": 0, "neither": 0}
    for clip, action, verb, noun in tqdm(loader, desc=f"eval({split})", leave=False):
        clip = clip.to(device); action = action.to(device)
        verb = verb.to(device); noun = noun.to(device)
        v_lg = verb_model(clip)
        n_lg = noun_model(clip)
        a_lg = action_logits_from_vn(v_lg, n_lg, device)
        bs = clip.size(0); n += bs
        ca = topk_acc(a_lg, action, a_K)
        for k in a_K: a_c[k] += ca[k]
        pv = v_lg.argmax(1); pn = n_lg.argmax(1)
        v_ok = (pv == verb); n_ok = (pn == noun)
        cell["both"]      += int((v_ok & n_ok).sum())
        cell["verb_only"] += int((v_ok & ~n_ok).sum())
        cell["noun_only"] += int((~v_ok & n_ok).sum())
        cell["neither"]   += int((~v_ok & ~n_ok).sum())
    return {"n": n,
            **{f"action_top{k}": a_c[k]/n*100 for k in a_K},
            "err_decomp": {k: cell[k]/n*100 for k in cell}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--view", required=True, choices=VIEWS)
    p.add_argument("--hand", required=True, choices=HANDS)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_action_lookup(device)

    v_ckpt = PROJECT_DIR / "checkpoints" / f"impact_compvar_verb_only_{args.view}_{args.hand}" / "best.pth"
    n_ckpt = PROJECT_DIR / "checkpoints" / f"impact_compvar_noun_only_{args.view}_{args.hand}" / "best.pth"
    if not v_ckpt.exists() or not n_ckpt.exists():
        print(f"[error] missing checkpoint(s): {v_ckpt.exists()=} {n_ckpt.exists()=}")
        return

    verb_model = SingleHeadModel(N_VERB, dropout=0.0, pretrained=False).to(device)
    cv = torch.load(v_ckpt, map_location=device, weights_only=False)
    verb_model.load_state_dict(cv["model"])
    noun_model = SingleHeadModel(N_NOUN, dropout=0.0, pretrained=False).to(device)
    cn = torch.load(n_ckpt, map_location=device, weights_only=False)
    noun_model.load_state_dict(cn["model"])
    print(f"[loaded] verb_only epoch={cv['epoch']}  noun_only epoch={cn['epoch']}")

    seen_ds   = IMPACTCompDataset(args.view, args.hand, "comp_val_seen", is_train=False)
    unseen_ds = IMPACTCompDataset(args.view, args.hand, "comp_val_unseen", is_train=False)
    dl = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True,
                               persistent_workers=False, prefetch_factor=2)
    seen = evaluate(verb_model, noun_model, dl(seen_ds), device, "seen")
    unseen = evaluate(verb_model, noun_model, dl(unseen_ds), device, "unseen")
    final_hm = hm(seen["action_top1"], unseen["action_top1"])
    print(f"SEEN  : t1={seen['action_top1']:.2f}% t5={seen['action_top5']:.2f}%")
    print(f"UNSEEN: t1={unseen['action_top1']:.2f}% t5={unseen['action_top5']:.2f}%")
    print(f"HM(action_top1) = {final_hm:.2f}")

    out_dir = PROJECT_DIR / "results" / f"impact_indep_{args.view}_{args.hand}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comp_eval.json", "w") as f:
        json.dump({"view": args.view, "hand": args.hand, "arch": "independent",
                   "verb_ckpt": str(v_ckpt), "noun_ckpt": str(n_ckpt),
                   "seen": seen, "unseen": unseen, "HM_action_top1": final_hm}, f, indent=2)
    print(f"[done] -> {out_dir/'comp_eval.json'}")


if __name__ == "__main__":
    main()
