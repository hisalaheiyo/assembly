"""
Combine independently-trained verb + noun models on MECCANO comp split,
reconstruct action_id from (verb, noun) outputs, and report top-K accuracy.

Usage:
  python eval_meccano_indep_action.py
"""
import sys, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_meccano_comp import (
    MeccanoCompDataset, IndepHeadModel,
    action_logits_from_vn, build_action_lookup,
    N_VERB, N_NOUN, N_ACTION, COMP_DIR,
)

from paths import PROJECT_DIR  # see code/paths.py for layout / env-var overrides
A_K = [1, 3, 5, 10]
V_K = [1, 3, 5]
N_K = [2, 3, 5, 10]


def topk_acc(logits, target, K_list):
    maxk = min(max(K_list), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(target.unsqueeze(1).expand_as(pred))
    out = {}
    for k in K_list:
        kk = min(k, logits.size(1))
        out[k] = correct[:, :kk].any(dim=1).float().sum().item()
    return out


def topk_coverage(logits, target_vec, K):
    K_eff = min(K, logits.size(1))
    _, top = logits.topk(K_eff, dim=1)
    correct = 0
    for b in range(logits.size(0)):
        gt = (target_vec[b] > 0).nonzero(as_tuple=True)[0]
        if len(gt) == 0: correct += 1
        else:
            ts = set(top[b].tolist())
            if all(g.item() in ts for g in gt): correct += 1
    return correct


@torch.no_grad()
def evaluate(verb_model, noun_model, loader, device):
    verb_model.eval(); noun_model.eval()
    n = 0
    a_c = {k: 0 for k in A_K}; v_c = {k: 0 for k in V_K}; n_cov = {k: 0 for k in N_K}
    for clip, action, verb, noun_vec in loader:
        clip, action, verb, noun_vec = (clip.to(device), action.to(device),
                                         verb.to(device), noun_vec.to(device))
        v_lg = verb_model(clip)
        n_lg = noun_model(clip)
        a_lg = action_logits_from_vn(v_lg, n_lg, device)
        bs = clip.size(0); n += bs
        ca = topk_acc(a_lg, action, A_K); cv = topk_acc(v_lg, verb, V_K)
        for k in A_K: a_c[k] += ca[k]
        for k in V_K: v_c[k] += cv[k]
        for k in N_K: n_cov[k] += topk_coverage(n_lg, noun_vec, k)
    return {"n": n,
            **{f"action_top{k}": a_c[k]/n*100 for k in A_K},
            **{f"verb_top{k}": v_c[k]/n*100 for k in V_K},
            **{f"noun_cov{k}": n_cov[k]/n*100 for k in N_K}}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load both models
    verb_ckpt = PROJECT_DIR / "checkpoints" / "meccano_comp_indep_verb" / "best.pth"
    noun_ckpt = PROJECT_DIR / "checkpoints" / "meccano_comp_indep_noun" / "best.pth"
    if not verb_ckpt.exists() or not noun_ckpt.exists():
        print(f"[error] checkpoints missing")
        return

    verb_m = IndepHeadModel("verb", dropout=0.0, pretrained=False)
    cv = torch.load(verb_ckpt, map_location=device, weights_only=False)
    verb_m.load_state_dict(cv["model"])
    verb_m.to(device).eval()
    print(f"[verb] loaded epoch={cv['epoch']}, best={cv.get('best',0):.2f}")

    noun_m = IndepHeadModel("noun", dropout=0.0, pretrained=False)
    cn = torch.load(noun_ckpt, map_location=device, weights_only=False)
    noun_m.load_state_dict(cn["model"])
    noun_m.to(device).eval()
    print(f"[noun] loaded epoch={cn['epoch']}, best={cn.get('best',0):.2f}")

    build_action_lookup(device)

    # Eval on val_seen and val_unseen
    results = {}
    for split in ["comp_val_seen", "comp_val_unseen"]:
        ds = MeccanoCompDataset(split, is_train=False)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=8, pin_memory=True)
        res = evaluate(verb_m, noun_m, loader, device)
        sp_label = "val_seen" if split == "comp_val_seen" else "val_unseen"
        results[sp_label] = res
        print(f"\n{sp_label}:")
        print(f"  action: t1={res['action_top1']:.2f}% t3={res['action_top3']:.2f}% "
              f"t5={res['action_top5']:.2f}% t10={res['action_top10']:.2f}%")
        print(f"  verb:   t1={res['verb_top1']:.2f}% t3={res['verb_top3']:.2f}% t5={res['verb_top5']:.2f}%")
        print(f"  noun:   c2={res['noun_cov2']:.2f}% c3={res['noun_cov3']:.2f}% "
              f"c5={res['noun_cov5']:.2f}% c10={res['noun_cov10']:.2f}%")

    out_dir = PROJECT_DIR / "results" / "meccano_comp_indep_combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "final_eval.json", "w") as f:
        json.dump({"arch": "indep_combined",
                   "verb_ckpt": str(verb_ckpt), "noun_ckpt": str(noun_ckpt),
                   **results}, f, indent=2)
    print(f"\n[done] saved to {out_dir/'final_eval.json'}")


if __name__ == "__main__":
    main()
