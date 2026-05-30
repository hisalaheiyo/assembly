"""
D1b: joint (shared) vs solo (independent) per-component seen→unseen degradation,
across all 3 datasets. The "extra degradation" of joint over solo reveals the
verb/noun head's hidden context-dependence (the shortcut effect).

Reads only existing comp_eval.json files; produces no models. CPU-only.

Output: results/d1b_summary.json
"""

import json
from pathlib import Path
from paths import PROJECT_DIR  # see code/paths.py for layout / env-var overrides
RESULTS = PROJECT_DIR / "results"


def read_json(p):
    return json.load(open(p)) if Path(p).exists() else None


def get(d, *path, default=None):
    for k in path:
        if d is None or k not in d: return default
        d = d[k]
    return d


def meccano():
    """MECCANO: shared (joint) vs indep_verb/noun (solo). Multi-label noun → use
    verb head only for clean D1b. Noun side has the multi-label complication."""
    j = read_json(RESULTS / "meccano_comp_shared/final_eval.json")
    sv = read_json(RESULTS / "meccano_comp_indep_verb/final_eval.json")
    sn = read_json(RESULTS / "meccano_comp_indep_noun/final_eval.json")
    out = {}
    if j and sv:
        joint_v_seen   = get(j,  "val_seen",   "verb_top1")
        joint_v_unseen = get(j,  "val_unseen", "verb_top1")
        solo_v_seen    = get(sv, "val_seen",   "verb_top1")
        solo_v_unseen  = get(sv, "val_unseen", "verb_top1")
        out["verb"] = compute(joint_v_seen, joint_v_unseen, solo_v_seen, solo_v_unseen)
    if j and sn:
        # Noun: use noun_cov3 since shared/indep MECCANO noun is multi-label BCE
        joint_n_seen   = get(j,  "val_seen",   "noun_cov3")
        joint_n_unseen = get(j,  "val_unseen", "noun_cov3")
        solo_n_seen    = get(sn, "val_seen",   "noun_cov3")
        solo_n_unseen  = get(sn, "val_unseen", "noun_cov3")
        out["noun"] = compute(joint_n_seen, joint_n_unseen, solo_n_seen, solo_n_unseen)
    return out


def impact():
    """IMPACT: comp multihead (joint) vs verb_only/noun_only (solo), per view×hand."""
    VIEWS = ["front", "left", "right", "top", "ego"]
    HANDS = ["left", "right"]
    rows = []
    for view in VIEWS:
        for hand in HANDS:
            j = read_json(RESULTS / f"impact_comp_{view}_{hand}/comp_eval.json")
            sv = read_json(RESULTS / f"impact_compvar_verb_only_{view}_{hand}/comp_eval.json")
            sn = read_json(RESULTS / f"impact_compvar_noun_only_{view}_{hand}/comp_eval.json")
            row = {"view": view, "hand": hand}
            if j and sv:
                row["verb"] = compute(get(j,"seen","verb_top1"), get(j,"unseen","verb_top1"),
                                       get(sv,"seen_top1"), get(sv,"unseen_top1"))
            if j and sn:
                row["noun"] = compute(get(j,"seen","noun_top1"), get(j,"unseen","noun_top1"),
                                       get(sn,"seen_top1"), get(sn,"unseen_top1"))
            if "verb" in row or "noun" in row:
                rows.append(row)
    return rows


def havid():
    """HAViD: havid_gran_k2 (joint 2-head) vs solo_verb/solo_manip, per subtask."""
    SUBT = ["view0_lh_aa","view0_rh_aa","view1_lh_aa","view1_rh_aa","view2_lh_aa","view2_rh_aa"]
    rows = []
    for st in SUBT:
        j = read_json(RESULTS / f"havid_gran_k2_{st}/comp_eval.json")
        sv = read_json(RESULTS / f"havid_gran_solo_verb_{st}/comp_eval.json")
        sm = read_json(RESULTS / f"havid_gran_solo_manip_{st}/comp_eval.json")
        row = {"subtask": st}
        # havid_gran k2 saves per-head: seen/unseen with verb_top1 / manip_top1 keys
        if j and sv:
            row["verb"] = compute(get(j,"seen","verb_top1"), get(j,"unseen","verb_top1"),
                                   get(sv,"seen","joint_top1"), get(sv,"unseen","joint_top1"))
        if j and sm:
            row["manip"] = compute(get(j,"seen","manip_top1"), get(j,"unseen","manip_top1"),
                                    get(sm,"seen","joint_top1"), get(sm,"unseen","joint_top1"))
        if "verb" in row or "manip" in row:
            rows.append(row)
    return rows


def compute(j_s, j_u, s_s, s_u):
    if j_s is None or j_u is None or s_s is None or s_u is None:
        return None
    joint_deg = j_s - j_u
    solo_deg = s_s - s_u
    return {
        "joint_seen":  round(j_s, 2), "joint_unseen": round(j_u, 2),
        "solo_seen":   round(s_s, 2), "solo_unseen":  round(s_u, 2),
        "joint_seen_unseen_drop": round(joint_deg, 2),
        "solo_seen_unseen_drop":  round(solo_deg, 2),
        "shortcut_diff_pp": round(joint_deg - solo_deg, 2),
        "D1a_seen_margin_pp": round(j_s - s_s, 2),
    }


def main():
    out = {
        "meccano": meccano(),
        "impact_per_config": impact(),
        "havid_per_subtask": havid(),
    }
    # IMPACT and HAViD aggregates (average across views/subtasks)
    def avg(lst, key, sub):
        vals = [r[key][sub] for r in lst if r.get(key) and r[key].get(sub) is not None]
        return round(sum(vals)/len(vals), 2) if vals else None

    out["impact_avg"] = {
        "verb": {k: avg(out["impact_per_config"], "verb", k)
                 for k in ["joint_seen_unseen_drop", "solo_seen_unseen_drop",
                           "shortcut_diff_pp", "D1a_seen_margin_pp"]},
        "noun": {k: avg(out["impact_per_config"], "noun", k)
                 for k in ["joint_seen_unseen_drop", "solo_seen_unseen_drop",
                           "shortcut_diff_pp", "D1a_seen_margin_pp"]},
    }
    out["havid_avg"] = {
        "verb": {k: avg(out["havid_per_subtask"], "verb", k)
                 for k in ["joint_seen_unseen_drop", "solo_seen_unseen_drop",
                           "shortcut_diff_pp", "D1a_seen_margin_pp"]},
        "manip": {k: avg(out["havid_per_subtask"], "manip", k)
                  for k in ["joint_seen_unseen_drop", "solo_seen_unseen_drop",
                            "shortcut_diff_pp", "D1a_seen_margin_pp"]},
    }

    out_path = RESULTS / "d1b_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_path}")

    # Pretty-print the headline rows
    print(f"\n{'='*70}\nD1b RESULTS (shortcut margin: joint − solo seen→unseen degradation)\n{'='*70}")
    print(f"{'dataset/config':<32}{'head':<8}{'joint_drop':>12}{'solo_drop':>12}{'shortcut Δ':>13}")
    if out["meccano"]:
        for h, d in out["meccano"].items():
            if d: print(f"{'MECCANO':<32}{h:<8}{d['joint_seen_unseen_drop']:>12}"
                        f"{d['solo_seen_unseen_drop']:>12}{d['shortcut_diff_pp']:>13}")
    for h in ["verb","noun"]:
        d = out["impact_avg"][h]
        if d["joint_seen_unseen_drop"] is not None:
            print(f"{'IMPACT (avg)':<32}{h:<8}{d['joint_seen_unseen_drop']:>12}"
                  f"{d['solo_seen_unseen_drop']:>12}{d['shortcut_diff_pp']:>13}")
    for h in ["verb","manip"]:
        d = out["havid_avg"][h]
        if d["joint_seen_unseen_drop"] is not None:
            print(f"{'HAViD (avg)':<32}{h:<8}{d['joint_seen_unseen_drop']:>12}"
                  f"{d['solo_seen_unseen_drop']:>12}{d['shortcut_diff_pp']:>13}")


if __name__ == "__main__":
    main()
