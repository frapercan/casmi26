"""Evaluation registry (scripts/meta/registry.py), version-2 explorations, in throwaway sandboxes: fake cells, library,
weights, eval files and a git repository under tmp_path; the real registry/ and work/ are never touched."""
import copy, json, os, shutil, sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import registry_dryrun as dr  # noqa: E402

C2 = [f"np_c2_{i:04d}" for i in range(5)]
NIT = [f"np_nit_{i:04d}" for i in range(5)]


@pytest.fixture
def sb(tmp_path, monkeypatch):
    """Sandbox: cwd inside it, git confined to it, a fresh registry module (empty caches) reading the fake holdout."""
    root = tmp_path / "sb"
    root.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.delenv("CASMI_CELL_DIRS", raising=False)
    monkeypatch.chdir(root)
    reg = dr.load_registry()
    reg.HOLDOUT = str(root / "holdout.parquet")
    return SimpleNamespace(root=str(root), reg=reg)


def world(sb, weights, n=(("np_nit", 5), ("np_c2", 5), ("np_tims", 4))):
    cells = dr.build_world(sb.root, n=n, weights=weights)
    sb.reg.main(["cells", "freeze"])
    dr.git_init(sb.root)
    dr.commit_all(sb.root, "world")
    return cells


def put(sb, member, cell, mols, **cols):
    pd.DataFrame({"mol_id": mols, **cols}).to_parquet(f"{sb.root}/work/cells/eval/{member}__{cell}.parquet", index=False)


def put_arm(sb, members, cell, **stats):
    json.dump(stats, open(f"{sb.root}/work/cells/eval/{'+'.join(members)}__{cell}.arm.json", "w"))


def draft(eid="E901", **over):
    E = {"id": eid, "title": "sandbox", "kind": "one_axis", "question": "q", "hypothesis": "h",
         "arms": {"A0": ["a0s0", "a0s1"], "A1": ["a1s0", "a1s1"]}, "candidate_arms": ["A1"], "cells": ["np_c2"],
         "arms_with_results_at_freeze": [],
         "rules": {"version": 2, "metrics": ["p25:nf"], "arm_metrics": ["lift_fe"], "contrasts": [["A1", "A0"]],
                   "bootstrap": {"resamples": 500, "seed": 3, "level": 0.95}, "min_seeds": 1,
                   "verdict": {"V": [{"contrast": ["A1", "A0"], "cell": "np_c2", "metric": "p25:nf", "test": "mean_gt", "value": 0}]}}}
    rules = over.pop("rules", {})
    E.update(over)
    E["rules"].update(rules)
    return E


def freeze(sb, E):
    p = f"{sb.root}/draft_{E['id']}.json"
    json.dump(E, open(p, "w"))
    sb.reg.main(["explore", "freeze", p])
    return json.load(open(f"{sb.root}/registry/explorations/{E['id']}.json"))


def report(sb, E, capsys=None):
    freeze(sb, E)
    dr.commit_all(sb.root, f"freeze {E['id']}")
    sb.reg.main(["explore", "report", E["id"]])
    return json.load(open(f"{sb.root}/registry/explorations/{E['id']}.json"))


def boot_ref(d, w, seed, B):
    """The documented bootstrap, written out independently: molecules in mol_id order, B resamples, fresh generator."""
    idx = np.random.default_rng(seed).integers(len(d), size=(B, len(d)))
    return d[idx].mean(1) if w is None else (w[idx] * d[idx]).sum(1) / w[idx].sum(1)


# ------------------------------------------------------------------------------------------------------------ freeze

def test_freeze_keeps_declared_arms_with_results(sb):
    world(sb, ["a0s0", "a0s1", "a1s0", "a1s1"])
    put(sb, "a0s0", "np_c2", C2[:2], **{"p25:nf": [1.0, 0.5]})
    put(sb, "a0s1~mir", "np_c2", C2[:2], **{"p25:nf": [1.0, 0.5]})
    put_arm(sb, ["a0s0", "a0s1"], "np_c2", lift_fe={"est": 0.2, "n": 5})
    declared = ["A0 (a0s0 only; np_c2, read while drafting)"]
    arms = {"A0": ["a0s0", "a0s1"], "A1": ["a1s0", "a1s1"], "A0r": ["a0s0~mir", "a0s1~mir"]}
    E = freeze(sb, draft(arms=arms, arms_with_results_at_freeze=declared, post_hoc=False))
    assert E["arms_with_results_at_freeze"] == declared
    assert E["arms_with_results_detected"] == {"A0": {"members": {"a0s0": ["np_c2"]}, "arm_files": ["a0s0+a0s1__np_c2.arm.json"]},
                                               "A0r": {"members": {"a0s1~mir": ["np_c2"]}}}
    assert E["post_hoc"] is False and E["state"] == "frozen"
    assert E["frozen_sha256"] == sb.reg.frozen_sha(E) and E["rules_sha256"] == sb.reg.sha_obj(E["rules"])
    assert set(E["cell_manifests"]) == {"np_c2"}
    # a candidate arm with results, detected but not declared: post_hoc, the declaration is still kept as written
    put(sb, "a1s1", "np_c2", C2[:2], **{"p25:nf": [1.0, 0.5]})
    E2 = freeze(sb, draft("E902", arms=arms, arms_with_results_at_freeze=declared))
    assert E2["arms_with_results_at_freeze"] == declared and E2["post_hoc"] is True
    assert E2["arms_with_results_detected"]["A1"] == {"members": {"a1s1": ["np_c2"]}}
    os.remove(f"{sb.root}/work/cells/eval/a1s1__np_c2.parquet")
    # a candidate arm only declared (no file left): post_hoc too
    E3 = freeze(sb, draft("E903", arms=arms, arms_with_results_at_freeze=["a1s0 (np_c2 numbers seen in a notebook)"]))
    assert E3["post_hoc"] is True and "A1" not in E3["arms_with_results_detected"]


def test_freeze_refuses_duplicate_reserved_and_unfrozen(sb):
    world(sb, ["a0s0", "a0s1", "a1s0", "a1s1"])
    with pytest.raises(SystemExit, match=r"E006 is reserved .*FragWM draft .* also used E006.*next free ID is E013"):
        freeze(sb, draft("E006", title="FragWM: fragment world model"))
    assert freeze(sb, draft("E006", slug="sr1-neutral-negatives"))["slug"] == "sr1-neutral-negatives"
    freeze(sb, draft("E901"))
    with pytest.raises(SystemExit, match="already frozen"):
        freeze(sb, draft("E901", title="another one"))
    dr.commit_all(sb.root, "freeze E901")
    os.remove(f"{sb.root}/registry/explorations/E901.json")
    dr.git(sb.root, "add", "-A", "registry")
    dr.git(sb.root, "commit", "-q", "-m", "E901 deleted (never do this)")
    with pytest.raises(SystemExit, match="existed in git history"):
        freeze(sb, draft("E901"))
    with pytest.raises(SystemExit, match=r"cells \['np_like'\] are not frozen"):
        freeze(sb, draft("E904", cells=["np_c2", "np_like"]))
    v1 = draft("E905")
    v1["rules"].pop("version")
    with pytest.raises(SystemExit, match="rules.version must be 2"):
        freeze(sb, v1)
    with pytest.raises(SystemExit, match="id must look like E006"):
        freeze(sb, draft("X1"))


def cond(**kw):
    c = {"contrast": ["A1", "A0"], "cell": "np_c2", "metric": "p25:nf", "test": "ci_low_gt", "value": 0}
    c.update(kw)
    return {k: v for k, v in c.items() if v is not None}


BAD = {
    "non-list block": (lambda E: E["rules"]["verdict"].update(S="A2: identical blocks with A2 in place of A1"), "is a str"),
    "bare condition": (lambda E: E["rules"]["verdict"].update(S=cond()), "single condition still goes inside a list"),
    "empty block": (lambda E: E["rules"]["verdict"].update(S=[]), "empty block"),
    "unknown test": (lambda E: E["rules"]["verdict"].update(S=[cond(test="mean>-0.03")]),
                     "unknown test 'mean>-0.03' (one of mean_gt, mean_lt, abs_mean_lt, ci_low_gt, ci_high_lt, abs_ci_lt); "
                     'version 2 writes "test": "mean_gt", "value": -0.03'),
    "unknown test name": (lambda E: E["rules"]["verdict"].update(S=[cond(test="mean_ge")]), "unknown test 'mean_ge'"),
    "unknown arm": (lambda E: E["rules"]["verdict"].update(S=[cond(contrast=["A9", "A0"])]), "unknown arm 'A9'"),
    "same arm twice": (lambda E: E["rules"]["verdict"].update(S=[cond(contrast=["A0", "A0"])]), "two different arms"),
    "unknown cell": (lambda E: E["rules"]["verdict"].update(S=[cond(cell="np_tims")]), "cell 'np_tims' is neither"),
    "undeclared pseudo-cell": (lambda E: E["rules"]["verdict"].update(
        S=[cond(contrast=None, arm="A1", cell="np_pool", metric="lift_fe", test="abs_mean_lt", value=0.1)]), "cell 'np_pool' is neither"),
    "pseudo-cell, per-molecule metric": (lambda E: (E.update(pseudo_cells={"np_pool": ["np_c2", "np_tims"]}),
                                                    E["rules"]["verdict"].update(S=[cond(contrast=None, arm="A1", cell="np_pool")])),
                                         "holds arm-level metrics only"),
    "pseudo-cell shadows a cell": (lambda E: E.update(pseudo_cells={"np_c2": ["np_tims"]}), "shadows a cell"),
    "unknown metric": (lambda E: E["rules"]["verdict"].update(S=[cond(metric="p25:nce")]), "unknown metric 'p25:nce'"),
    "arm-level metric in a contrast": (lambda E: E["rules"]["verdict"].update(S=[cond(metric="lift_fe")]), "test it per arm"),
    "weights on an arm-level metric": (lambda E: E["rules"]["verdict"].update(
        S=[cond(contrast=None, arm="A1", metric="lift_fe", weights="w_mass")]), "weights apply to per-molecule"),
    "unknown composite name": (lambda E: E["rules"]["verdict"].update(X={"all_of": ["V", "S9"]}), "unknown verdict name 'S9'"),
    "composite cycle": (lambda E: E["rules"]["verdict"].update(X={"all_of": ["V", {"not": "Y"}]}, Y={"any_of": ["X"]}), "form a cycle"),
    "self reference": (lambda E: E["rules"]["verdict"].update(X={"not": "X"}), "form a cycle"),
    "bad composite": (lambda E: E["rules"]["verdict"].update(X={"all_of": ["V"], "any_of": ["V"]}), "a composite is"),
    "one-sided side mismatch": (lambda E: E["rules"]["verdict"].update(S=[cond(sided="upper")]), "cannot use a upper bound"),
    "abs_ci one-sided": (lambda E: E["rules"]["verdict"].update(S=[cond(test="abs_ci_lt", value=0.1, sided="lower")]), "cannot use a lower bound"),
    "level on a mean test": (lambda E: E["rules"]["verdict"].update(S=[cond(test="mean_gt", level=0.9)]), "level and sided do not apply"),
    "missing value": (lambda E: E["rules"]["verdict"].update(S=[cond(value=None)]), "value must be a number"),
    "bad level": (lambda E: E["rules"]["verdict"].update(S=[cond(level=95)]), "level must be in"),
    "typo key": (lambda E: E["rules"]["verdict"].update(S=[cond(tset="mean_gt")]), "unknown keys"),
    "reference arm in a verdict": (lambda E: (E.update(reference_arms={"A0_3060": ["w4base_s0"]}),
                                              E["rules"]["verdict"].update(S=[cond(contrast=["A1", "A0_3060"])])), "is a reference arm"),
    "reference arm as candidate": (lambda E: E.update(reference_arms={"A0_3060": ["w4base_s0"]}, candidate_arms=["A0_3060"]),
                                   "reference arms are never candidates"),
    "arm_weights": (lambda E: E.update(arm_weights={"A0": "a0s0"}), "arm_weights is version-1"),
    "string arm": (lambda E: E["arms"].update(A0="a0s0,a0s1"), "members must be a non-empty list"),
    "bad member name": (lambda E: E["arms"].update(A0=["a0s0+a0s1"]), "bad member name"),
    "undeclared results field": (lambda E: E.pop("arms_with_results_at_freeze"), "declare arms_with_results_at_freeze"),
    "declared entry names nothing": (lambda E: E.update(arms_with_results_at_freeze=["Z9 (np_c2)"]), "names no arm or member"),
    "contrast of unknown arm": (lambda E: E["rules"].update(contrasts=[["A1", "A7"]]), "rules.contrasts[0]"),
    "arm_stats unknown cell": (lambda E: E["rules"].update(arm_stats=[{"arm": "A1", "cell": "np_x", "metric": "lift_fe"}]), "'np_x' is neither"),
    "bootstrap level": (lambda E: E["rules"]["bootstrap"].update(level=1.5), "rules.bootstrap.level"),
    "min_seeds": (lambda E: E["rules"].update(min_seeds=0), "min_seeds must be a positive integer"),
    "draft with a verdict": (lambda E: E.update(verdict={"V": True}), "a draft carries no report or archive fields"),
    "list as cell": (lambda E: E["rules"]["verdict"].update(S=[cond(cell=["np_c2"])]), "cell and metric are names"),
    "list in contrast": (lambda E: E["rules"]["verdict"].update(S=[cond(contrast=[["A1"], "A0"])]), "'contrast' is [arm, arm]"),
    "list as test": (lambda E: E["rules"]["verdict"].update(S=[cond(test=["mean_gt"])]), "unknown test"),
    "list as candidate": (lambda E: E.update(candidate_arms=[["A1"]]), "must list arms"),
}


@pytest.mark.parametrize("case", sorted(BAD))
def test_validation_rejects_bad_rules(sb, case):
    E = draft()
    assert sb.reg.validate(E) == []
    mutate, expected = BAD[case]
    mutate(E)
    errors = sb.reg.validate(E)
    assert any(expected in e for e in errors), errors


def test_freeze_refuses_invalid_rules_with_every_error(sb):
    world(sb, ["a0s0", "a0s1", "a1s0", "a1s1"])
    E = draft()
    E["rules"]["verdict"].update(S="a sentence", T=[cond(test="mean>-0.03")])
    with pytest.raises(SystemExit) as e:
        freeze(sb, E)
    assert "cannot be frozen" in str(e.value) and "is a str" in str(e.value) and "unknown test" in str(e.value)
    assert not os.path.exists(f"{sb.root}/registry/explorations/E901.json")


# ------------------------------------------------------------------------------------------------------ statistics

def two_seed_world(sb):
    world(sb, ["a0s0", "a0s1", "a1s0", "a1s1", "a2s0", "a2s1", "a3s0"])
    put(sb, "a0s0", "np_c2", C2, **{"p25:nf": [0.5, 1.0, 0.2, 0.0, 0.25]})
    put(sb, "a0s1", "np_c2", C2, **{"p25:nf": [0.5, 0.5, 0.2, 0.1, np.nan]})
    put(sb, "a1s0", "np_c2", C2, **{"p25:nf": [1.0, 1.0, 0.5, 0.0, 0.25]})
    put(sb, "a1s1", "np_c2", [C2[i] for i in (0, 1, 2, 4)], **{"p25:nf": [1.0, 0.5, 0.25, 0.5]})


def test_contrast_seed_average_and_seed_sd_match_hand_numbers(sb):
    two_seed_world(sb)
    E = report(sb, draft())
    r = E["report"]
    c = r["contrasts"]["A1-A0|np_c2|p25:nf"]
    # A0 = [0.5, 0.75, 0.2, 0.05, 0.25] (np_c2_0004 from one seed), A1 = [1, 0.75, 0.375, 0.0 (one seed), 0.375]
    d = np.array([0.5, 0.0, 0.175, -0.05, 0.125])
    assert c["mean"] == pytest.approx(0.15) and c["n"] == 5 and (c["up"], c["down"]) == (3, 1)
    assert c["member_counts"] == {"A1": {"1": 1, "2": 4}, "A0": {"1": 1, "2": 4}}
    reps = boot_ref(d, None, 3, 500)
    assert c["ci"] == pytest.approx([np.percentile(reps, 2.5), np.percentile(reps, 97.5)], abs=1e-12)
    sd = r["seed_sd"]["A0"]["np_c2"]["p25:nf"]
    assert sd["mean_mol_sd"] == pytest.approx((0 + np.sqrt(0.125) + 0 + np.sqrt(0.005)) / 4)
    assert sd["sd_member_means"] == pytest.approx(np.std([0.425, 0.325], ddof=1)) and sd["n_common"] == 4
    assert E["verdict"] == {"V": True} and E["state"] == "reported"
    assert E["report_sha256"] == sb.reg.sha_obj(E["report"])


def test_weighted_mean_and_one_sided_bounds(sb):
    world(sb, ["a0s0", "a1s0", "a2s0"])
    w = [1.0, 2.0, 1.0, 4.0]
    put(sb, "a0s0", "np_nit", NIT[:4], **{"p25:nf": [0.2, 0.5, 1.0, 0.0], "w_mass": w})
    put(sb, "a1s0", "np_nit", NIT[:4], **{"p25:nf": [0.5, 0.5, 0.5, 0.25], "w_mass": w})
    put(sb, "a2s0", "np_nit", NIT[:4], **{"p25:nf": [0.5, 0.5, 0.5, 0.25], "w_mass": [1.0, 1.0, 1.0, 1.0]})
    base = {"contrast": ["A1", "A0"], "cell": "np_nit", "metric": "p25:nf", "weights": "w_mass"}
    V = {"S1": [dict(base, test="ci_low_gt", value=-0.5, level=0.95, sided="lower")],
         "up": [dict(base, test="ci_high_lt", value=0.5, level=0.9, sided="upper")],
         "two": [dict(base, test="abs_ci_lt", value=1.0, level=0.95)],
         "arm": [{"arm": "A1", "cell": "np_nit", "metric": "p25:nf", "weights": "w_mass", "test": "mean_gt", "value": 0.4}],
         "mixed": [dict(base, contrast=["A2", "A0"], test="mean_gt", value=0)]}
    E = report(sb, draft(arms={"A0": ["a0s0"], "A1": ["a1s0"], "A2": ["a2s0"]}, cells=["np_nit"],
                         rules={"contrasts": [["A1", "A0"]], "report_weights": {"np_nit": "w_mass"}, "verdict": V}))
    d, wv = np.array([0.3, 0.0, -0.5, 0.25]), np.array(w)
    reps = boot_ref(d, wv, 3, 500)
    C = E["report"]["conditions"]
    assert C["S1"][0]["estimate"] == pytest.approx(0.1)            # (0.3 - 0.5 + 4 * 0.25) / 8
    assert C["S1"][0]["bounds"] == [pytest.approx(np.percentile(reps, 5), abs=1e-12), None]
    assert C["up"][0]["bounds"] == [None, pytest.approx(np.percentile(reps, 90), abs=1e-12)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    assert C["two"][0]["bounds"] == pytest.approx([lo, hi], abs=1e-12)
    assert E["verdict"]["S1"] is bool(np.percentile(reps, 5) > -0.5) and E["verdict"]["two"] is bool(max(abs(lo), abs(hi)) < 1)
    assert C["arm"][0]["estimate"] == pytest.approx((0.5 + 1.0 + 0.5 + 1.0) / 8)
    cw = E["report"]["contrasts"]["A1-A0|np_nit|p25:nf|w:w_mass"]
    assert cw["mean"] == pytest.approx(0.1) and cw["n_eff"] == pytest.approx(64 / 22)
    assert E["report"]["contrasts"]["A1-A0|np_nit|p25:nf"]["mean"] == pytest.approx(0.0125)
    assert E["verdict"]["mixed"] == "undetermined" and "w_mass differs between arms" in C["mixed"][0]["why"]


def test_arm_level_statistics_boot_stored_ci_and_pseudo_cell(sb):
    world(sb, ["a0s0", "a0s1", "a1s0", "a1s1"])
    for m in ("a0s0", "a0s1", "a1s0", "a1s1"):
        put(sb, m, "np_c2", C2, **{"p25:nf": [0.5] * 5})
        put(sb, m, "np_tims", [f"np_tims_{i:04d}" for i in range(4)], **{"p25:nf": [0.5] * 4})
    boot = list(np.random.default_rng(1).normal(0.04, 0.03, 300))
    put_arm(sb, ["a1s0", "a1s1"], "np_c2", lift_fe={"est": 0.04, "boot": boot, "n": 5})
    put_arm(sb, ["a0s0", "a0s1"], "np_c2", lift_fe={"est": 0.27, "ci": {"0.95": [0.2, 0.33], "0.9": [0.21, 0.32]}, "n": 5})
    put_arm(sb, ["a1s0", "a1s1"], "np_pool", lift_fe_hidden={"est": 0.05, "ci": {"0.95": [-0.04, 0.14]}, "n": 9})
    a = lambda arm, **kw: dict({"arm": arm, "cell": "np_c2", "metric": "lift_fe"}, **kw)
    V = {"N1": [a("A1", test="abs_mean_lt", value=0.1), a("A1", test="ci_high_lt", value=0.15, level=0.95, sided="upper")],
         "abs": [a("A1", test="abs_ci_lt", value=0.2)],
         "stored_two": [a("A0", test="ci_low_gt", value=0.1)],
         "stored_upper": [a("A0", test="ci_high_lt", value=0.4, level=0.95, sided="upper")],
         "stored_lower975": [a("A0", test="ci_low_gt", value=0.1, level=0.975, sided="lower")],
         "stored_missing": [a("A0", test="ci_high_lt", value=0.4, level=0.99, sided="upper")],
         "N2": [{"arm": "A1", "cell": "np_pool", "metric": "lift_fe_hidden", "test": "abs_mean_lt", "value": 0.10}]}
    E = report(sb, draft(cells=["np_c2", "np_tims"], pseudo_cells={"np_pool": ["np_c2", "np_tims"]},
                         rules={"arm_metrics": ["lift_fe", "lift_fe_hidden"], "min_seeds": 2, "verdict": V,
                                "arm_stats": [{"arm": ["A0", "A1"], "cell": "np_c2", "metric": "lift_fe"},
                                              {"arm": "A1", "cell": "np_pool", "metric": "lift_fe_hidden"}]}))
    C, v, b = E["report"]["conditions"], E["verdict"], np.array(boot)
    assert C["N1"][1]["bounds"] == [None, pytest.approx(np.percentile(b, 95))] and v["N1"] is True
    lo, hi = np.percentile(b, [2.5, 97.5])
    assert C["abs"][0]["bounds"] == pytest.approx([lo, hi]) and v["abs"] is bool(max(abs(lo), abs(hi)) < 0.2)
    assert C["stored_two"][0]["bounds"] == [0.2, 0.33] and v["stored_two"] is True
    assert C["stored_upper"][0]["bounds"] == [None, 0.32] and v["stored_upper"] is True      # one-sided 95% = two-sided 90%
    assert C["stored_lower975"][0]["bounds"] == [0.2, None] and v["stored_lower975"] is True
    assert v["stored_missing"] == "undetermined" and "no stored 0.98 two-sided interval" in C["stored_missing"][0]["why"]
    assert v["N2"] is True and C["N2"][0]["members"] == {"A1": ["a1s0", "a1s1"]}
    S = E["report"]["arm_stats"]
    assert S["A1|np_c2|lift_fe"]["mean"] == 0.04 and S["A1|np_c2|lift_fe"]["ci"] == pytest.approx([lo, hi])
    assert S["A0|np_c2|lift_fe"]["ci"] == [0.2, 0.33] and S["A1|np_pool|lift_fe_hidden"]["ci"] == [-0.04, 0.14]
    assert E["report"]["results"]["A1"]["np_pool"]["members"] == ["a1s0", "a1s1"]


def test_missing_inputs_are_undetermined_never_true(sb):
    two_seed_world(sb)
    put(sb, "a2s0", "np_c2", C2, **{"p25:nf": [1.0] * 5})                       # A2: one seed of two finished
    put_arm(sb, ["a2s0"], "np_c2", lift_fe={"est": 0.0, "boot": [0.0, 0.01, -0.01], "n": 5})
    ok = {"contrast": ["A1", "A0"], "cell": "np_c2", "metric": "p25:nf", "test": "mean_gt", "value": 0}
    no = dict(ok, test="mean_lt")
    V = {"T": [ok], "F": [no],
         "min_seeds": [dict(ok, contrast=["A2", "A0"])],
         "min_seeds_arm": [{"arm": "A2", "cell": "np_c2", "metric": "lift_fe", "test": "abs_mean_lt", "value": 0.1}],
         "never_run": [dict(ok, contrast=["A3", "A0"])],
         "no_column": [dict(ok, metric="mrrT:nf")],
         "no_arm_file": [{"arm": "A1", "cell": "np_c2", "metric": "lift_fe", "test": "abs_mean_lt", "value": 0.1}],
         "T_and_missing": [ok, dict(ok, metric="mrrT:nf")],
         "F_and_missing": [no, dict(ok, metric="mrrT:nf")],
         "all_T_U": {"all_of": ["T", "never_run"]}, "all_F_U": {"all_of": ["F", "never_run"]},
         "any_T_U": {"any_of": ["T", "never_run"]}, "any_F_U": {"any_of": ["F", "never_run"]},
         "not_U": {"not": "never_run"}, "nested": {"all_of": ["T", {"not": "F"}, {"any_of": ["F", "T"]}]}}
    E = report(sb, draft(arms={"A0": ["a0s0", "a0s1"], "A1": ["a1s0", "a1s1"], "A2": ["a2s0", "a2s1"], "A3": ["a3s0"]},
                         rules={"metrics": ["p25:nf", "mrrT:nf", "pipe25:nf"], "min_seeds": 2, "verdict": V,
                                "contrasts": [["A1", "A0"], ["A2", "A0"], ["A3", "A0"]]}))
    v, C = E["verdict"], E["report"]["conditions"]
    assert (v["T"], v["F"]) == (True, False)
    for k in ("min_seeds", "min_seeds_arm", "never_run", "no_column", "no_arm_file", "T_and_missing", "all_T_U", "any_F_U", "not_U"):
        assert v[k] == "undetermined", k
    assert (v["F_and_missing"], v["all_F_U"], v["any_T_U"], v["nested"]) == (False, False, True, True)
    assert "A2 has 1 member(s) with results, min_seeds 2" in C["min_seeds"][0]["why"] and "estimate" in C["min_seeds"][0]
    assert "A2 has 1 member(s)" in C["min_seeds_arm"][0]["why"]
    assert "A3 has no results on np_c2" in C["never_run"][0]["why"]
    assert "no mrrT:nf column" in C["no_column"][0]["why"] and "missing" in C["no_arm_file"][0]["why"]
    R = E["report"]
    assert R["contrasts"]["A3-A0|np_c2|p25:nf"] == {"missing": "A3 has no results on np_c2"}
    assert "missing" in R["contrasts"]["A1-A0|np_c2|pipe25:nf"]
    assert R["contrasts"]["A2-A0|np_c2|p25:nf"]["below_min_seeds"] == ["A2"]
    assert any(m.startswith("A1-A0|np_c2|pipe25:nf") for m in R["missing"])
    assert R["results"]["A2"]["np_c2"] == {"members": ["a2s0"], "missing": ["a2s1"], "n_molecules": 5}


# ----------------------------------------------------------------------------------------------------- refusals

def test_report_refuses_uncommitted_or_changed_frozen_files(sb):
    two_seed_world(sb)
    freeze(sb, draft("E901"))
    with pytest.raises(SystemExit, match="is not committed"):
        sb.reg.main(["explore", "report", "E901"])
    dr.commit_all(sb.root, "freeze E901")
    p = f"{sb.root}/registry/explorations/E901.json"
    E = json.load(open(p))
    E["rules"]["verdict"]["V"][0]["value"] = -1
    json.dump(E, open(p, "w"), indent=1)
    with pytest.raises(SystemExit, match="differs from its committed version"):
        sb.reg.main(["explore", "report", "E901"])
    dr.commit_all(sb.root, "edit the frozen rules afterwards")
    with pytest.raises(SystemExit, match="differs from the commit that added it"):
        sb.reg.main(["explore", "report", "E901"])
    # edited before its first commit: git holds it, the hash catches it
    freeze(sb, draft("E902"))
    p = f"{sb.root}/registry/explorations/E902.json"
    E = json.load(open(p))
    E["hypothesis"] = "rewritten after the results"
    json.dump(E, open(p, "w"), indent=1)
    dr.commit_all(sb.root, "freeze E902")
    with pytest.raises(SystemExit, match="frozen object changed after freezing"):
        sb.reg.main(["explore", "report", "E902"])
    # a clean one reports once
    freeze(sb, draft("E903"))
    dr.commit_all(sb.root, "freeze E903")
    sb.reg.main(["explore", "report", "E903"])
    dr.commit_all(sb.root, "report E903")
    with pytest.raises(SystemExit, match="is 'reported'.*--again --reason"):
        sb.reg.main(["explore", "report", "E903"])
    with pytest.raises(SystemExit, match="--again needs a --reason"):
        sb.reg.main(["explore", "report", "E903", "--again"])
    first = json.load(open(f"{sb.root}/registry/explorations/E903.json"))
    sb.reg.main(["explore", "report", "E903", "--again", "--reason", "A1 second seed finished before the deadline"])
    E = json.load(open(f"{sb.root}/registry/explorations/E903.json"))
    assert E["report_history"] == [{"reported_at": first["reported_at"], "report_commit": first["report_commit"],
                                    "report_sha256": first["report_sha256"], "verdict": first["verdict"],
                                    "superseded_because": "A1 second seed finished before the deadline"}]
    assert E["frozen_sha256"] == sb.reg.frozen_sha(E) and E["report_sha256"] == first["report_sha256"]
    dr.commit_all(sb.root, "report E903 again")
    # the cell changed after the freeze
    freeze(sb, draft("E904"))
    dr.commit_all(sb.root, "freeze E904")
    m = pd.read_parquet(f"{sb.root}/work/cells/np_c2/molecules.parquet")
    m.assign(doc=m.doc + 1).to_parquet(f"{sb.root}/work/cells/np_c2/molecules.parquet", index=False)
    with pytest.raises(SystemExit, match="differs from the frozen manifest"):
        sb.reg.main(["explore", "report", "E904"])


def test_report_accepts_freeze_commit_that_removes_the_draft(sb):
    """git may see 'draft deleted + frozen file added' as a rename; the freeze commit is still found."""
    two_seed_world(sb)
    os.makedirs(f"{sb.root}/registry/proposals")
    p = f"{sb.root}/registry/proposals/E906.json"
    json.dump(draft("E906"), open(p, "w"), indent=1)
    dr.commit_all(sb.root, "draft E906")
    sb.reg.main(["explore", "freeze", p])
    os.remove(p)
    dr.commit_all(sb.root, "freeze E906, draft moved")
    sb.reg.main(["explore", "report", "E906"])
    assert json.load(open(f"{sb.root}/registry/explorations/E906.json"))["state"] == "reported"


def test_report_refuses_unregistered_weights_with_results(sb):
    two_seed_world(sb)
    put(sb, "ghost_s0", "np_c2", C2, **{"p25:nf": [0.0] * 5})
    with pytest.raises(SystemExit, match="ghost_s0 is not in registry/weights.jsonl"):
        report(sb, draft(arms={"A0": ["a0s0", "a0s1"], "A1": ["a1s0", "a1s1"], "A9": ["ghost_s0"]}))


# --------------------------------------------------------------------------------------------------------- leaks

def nit_world(sb):
    """np_nit molecules: 0 absent (a db negative of the molecule table), 1 a visible train structure, 2 hidden by the
    weights' exclusion list, 3 an official-key alias of a visible train structure, 4 visible through ik14_official only,
    5 np-examples and 6 holdout (always hidden), 7 a db negative."""
    world(sb, [dr.weights_row("wx"), dr.weights_row("wnone", exclude=None), dr.weights_row("wchild", init="wnone")])
    keys = ["NIT00000000000", "TRAIN000000000", "CTWO0000000000", "NIT00000000099", "NIT00000000098",
            "TIMS0000000000", "HOLD0000000000", "PCDB0000000000"]
    skey = list(keys)
    skey[3] = "OFFICIALALIAS1"
    official = [None] * 8
    official[4] = "TRAIN000000002"
    m = pd.DataFrame({"mol_id": [f"np_nit_{i:04d}" for i in range(8)], "ik14_lib": keys, "ik14_raw": keys, "skey": skey,
                      "ik14_official": official, "doc": 1.0, "exact_mass": 300.0})
    m.to_parquet(f"{sb.root}/work/cells/np_nit/molecules.parquet", index=False)
    meta = pd.read_parquet(f"{sb.root}/work/meta/v1/molecules.parquet")
    meta.loc[meta.ik14_lib == "TRAIN000000001", "skey"] = "OFFICIALALIAS1"
    meta.to_parquet(f"{sb.root}/work/meta/v1/molecules.parquet", index=False)


def test_np_nit_absent_leak_rule(sb, monkeypatch):
    nit_world(sb)
    L = sb.reg.leak_report("wx", verbose=False)
    assert L["np_nit"] == {"spectra_leak:wx": 3, "emb_db_exposure": 2, "leak_free": False}
    assert L["np_c2"] == {"spectra_leak:wx": 0, "anchor_leak": 0, "leak_free": True}          # hidden rule, unchanged
    assert sb.reg.leak_report("wx~mir", verbose=False, cells=["np_nit"]) == {"np_nit": L["np_nit"]}
    Ln = sb.reg.leak_report("wnone", verbose=False)
    assert Ln["np_nit"]["spectra_leak:wnone"] == 4 and Ln["np_c2"]["spectra_leak:wnone"] == 5
    assert sb.reg.leak_report("wchild", verbose=False, cells=["np_nit"])["np_nit"] == {
        "spectra_leak:wchild": 3, "spectra_leak:wnone": 4, "emb_db_exposure": 2, "leak_free": False}
    # only absent or always-hidden structures: leak-free, and database-negative presence stays a count
    m = pd.read_parquet(f"{sb.root}/work/cells/np_nit/molecules.parquet").iloc[[0, 2, 5, 6, 7]]
    alt = os.path.join(sb.root, "alt_np_nit")
    os.makedirs(alt)
    m.to_parquet(f"{alt}/molecules.parquet", index=False)
    monkeypatch.setenv("CASMI_CELL_DIRS", json.dumps({"np_nit": alt}))
    assert sb.reg.cell_dir("np_nit") == alt and sb.reg.cell_dir("np_c2") == "work/cells/np_c2"
    assert sb.reg.leak_report("wx", verbose=False, cells=["np_nit"])["np_nit"] == {
        "spectra_leak:wx": 0, "emb_db_exposure": 2, "leak_free": True}


def test_leak_voids_main_arm_and_flags_reference_arm(sb):
    world(sb, ["a0s0", "a1s0", dr.weights_row("wleak", exclude=None)])
    for m in ("a0s0", "a1s0", "wleak"):
        put(sb, m, "np_c2", C2, **{"p25:nf": [0.5, 1.0, 0.2, 0.0, 0.25]})
    base = dict(arms={"A0": ["a0s0"], "A1": ["a1s0"]}, rules={"contrasts": [["A1", "A0"], ["A0", "Aref"]]})
    E = report(sb, draft("E901", reference_arms={"Aref": ["wleak"]}, **copy.deepcopy(base)))
    assert E["state"] == "reported" and E["report"]["contrasts"]["A0-Aref|np_c2|p25:nf"]["leaky"] is True
    assert E["report"]["leaks"]["wleak"]["np_c2"]["leak_free"] is False
    bad = copy.deepcopy(base)
    bad["arms"]["A1"] = ["a1s0", "wleak"]
    bad["rules"]["contrasts"] = [["A1", "A0"]]
    E = report(sb, draft("E902", **bad))
    assert E["state"] == "void" and "member wleak (wleak) leaks on np_c2" in E["verdict"]


def test_pseudo_cell_leaks_are_checked_on_its_constituents(sb):
    """np_c2 is not one of the exploration's cells, only a constituent of np_pool: a leak there still voids."""
    world(sb, ["a0s0", "a1s0", dr.weights_row("wleak", exclude=None)])
    V = {"N2": [{"arm": "A1", "cell": "np_pool", "metric": "lift_fe_hidden", "test": "abs_mean_lt", "value": 0.1}],
         "S": [{"contrast": ["A1", "A0"], "cell": "np_nit", "metric": "p25:nf", "test": "mean_gt", "value": 0}]}
    E = draft("E901", arms={"A0": ["a0s0"], "A1": ["a1s0", "wleak"]}, cells=["np_nit"], pseudo_cells={"np_pool": ["np_c2", "np_tims"]},
              rules={"arm_metrics": ["lift_fe_hidden"], "verdict": V})
    assert sb.reg.real_cells(E) == ["np_nit", "np_c2", "np_tims"]
    F = freeze(sb, E)
    assert set(F["cell_manifests"]) == {"np_nit", "np_c2", "np_tims"}
    dr.commit_all(sb.root, "freeze E901")
    sb.reg.main(["explore", "report", "E901"])
    R = json.load(open(f"{sb.root}/registry/explorations/E901.json"))
    assert R["state"] == "void" and "leaks on np_c2" in R["verdict"] and R["leaks"]["wleak"]["np_nit"]["leak_free"] is True


# --------------------------------------------------------------------------------------------- version 1 and archive

def test_version1_explorations_load_and_report_as_before(sb):
    world(sb, ["w4base_s0", "w4jepa_v2_s0", "w4jepa_cf_s0"])
    os.makedirs(f"{sb.root}/registry/explorations", exist_ok=True)
    for i in range(1, 6):
        shutil.copy(os.path.join(dr.REPO, "registry", "explorations", f"E00{i}.json"), f"{sb.root}/registry/explorations/")
        E = json.load(open(f"{sb.root}/registry/explorations/E00{i}.json"))
        assert sb.reg.rules_version(E) == 1 and sb.reg.all_arms(E) and sb.reg.real_cells(E) == E["cells"]
    rng = np.random.default_rng(5)
    T = [f"np_tims_{i:04d}" for i in range(4)]
    for w in ("w4base_s0", "w4jepa_v2_s0", "w4jepa_cf_s0"):
        put(sb, w, "np_c2", C2, **{"mass:nf+0.25doc": rng.random(5), "mass:nf": rng.random(5)})
        put(sb, w, "np_tims", T, **{"mass:nf+0.25doc": rng.random(4), "mass:nf": rng.random(4)})
    dr.commit_all(sb.root, "version-1 explorations")
    for eid in ("E003", "E005"):
        with pytest.raises(SystemExit, match="is archived"):
            sb.reg.main(["explore", "report", eid])
    E0 = json.load(open(f"{sb.root}/registry/explorations/E001.json"))
    sb.reg.main(["explore", "report", "E001"])
    E = json.load(open(f"{sb.root}/registry/explorations/E001.json"))
    # the version-1 algorithm written out: one generator across contrasts, 2.5/97.5 percentiles, three test names
    R = E0["rules"]
    rng = np.random.default_rng(R["bootstrap"]["seed"])
    per = lambda w, c: pd.read_parquet(f"{sb.root}/work/cells/eval/{w}__{c}.parquet").set_index("mol_id")
    rep = {}
    for x, y in R["contrasts"]:
        for c in E0["cells"]:
            A, B = per(E0["arms"][x], c), per(E0["arms"][y], c)
            for m in R["metrics"]:
                d = (A[m] - B.reindex(A.index)[m]).dropna().to_numpy(float)
                bs = d[rng.integers(len(d), size=(R["bootstrap"]["resamples"], len(d)))].mean(1)
                rep[f"{x}-{y}|{c}|{m}"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                         "n": len(d), "up": int((d > 0).sum()), "down": int((d < 0).sum())}
    assert E["report"] == rep and E["state"] == "reported"
    t = lambda c: {"ci_low>0": rep[k(c)]["ci95"][0] > 0, "mean>-0.01": rep[k(c)]["mean"] > -0.01, "ci_high<0": rep[k(c)]["ci95"][1] < 0}[c["test"]]
    k = lambda c: f"{c['contrast'][0]}-{c['contrast'][1]}|{c['cell']}|{c['metric']}"
    assert E["verdict"] == {n: all(t(c) for c in cs) for n, cs in R["verdict"].items()}


def test_archive_version2(sb):
    two_seed_world(sb)
    freeze(sb, draft("E901", arms={"A0": ["a0s0", "a0s1"], "A1": ["a1s0", "a1s1"], "A0r": ["a0s0~mir"]}))
    sb.reg.main(["explore", "archive", "E901", "--reason", "superseded before any result"])
    E = json.load(open(f"{sb.root}/registry/explorations/E901.json"))
    assert E["state"] == "archived" and E["results_existed"] is True
    assert E["frozen_sha256"] == sb.reg.frozen_sha(E)                      # archiving adds post-freeze fields only


# ------------------------------------------------------------------------------------------------------- dry run

def test_e006_like_dry_run(tmp_path, monkeypatch):
    monkeypatch.delenv("CASMI_CELL_DIRS", raising=False)
    root, E = dr.main(str(tmp_path / "dry"))
    v = E["verdict"]
    assert E["state"] == "reported" and E["post_hoc"] is False
    assert E["arms_with_results_at_freeze"] == ["A0_3060 (w4base_s0; np_c2 and np_tims, before this exploration)"]
    assert list(E["arms_with_results_detected"]) == ["A0_3060"]
    assert v["N1_A1"] is True and v["N2_A1"] is True and v["A1_admit"] is True
    assert v["N2_A2"] == "undetermined" and v["A2_admit"] == "undetermined" and v["A0r_ship"] is False
    R = E["report"]
    assert all(R["leaks"][w]["np_nit"] == {"spectra_leak:" + w: 0, "emb_db_exposure": 3, "leak_free": True} for w in R["leaks"])
    assert R["results"]["A2"]["np_tims"]["missing"] == ["w4mnt_t4s1"]
    assert "A0r|np_c2|lift_fe" in R["arm_stats"] and "A0-A0_3060|np_c2|p25:nf" in R["contrasts"]
    assert R["contrasts"]["A0-A0_3060|np_nit|p25:nf"] == {"missing": "A0_3060 has no results on np_nit"}
    assert any("pipe25:nf" in m for m in R["missing"])
