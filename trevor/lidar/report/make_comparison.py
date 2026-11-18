#!/usr/bin/env python
"""Cross-lane comparison figures + tables (LiDAR-only vs camera YOLO11s vs BEVFusion camera+LiDAR), nuScenes mini-val.

Inputs (all in lidar/report/inputs/ unless noted; teammates' files copied verbatim, see PROVENANCE.txt):
  fusion_summary.csv                 Christian, BEVFusion C+L, 38 conditions, mAP/NDS
  camera_accuracy_latency_summary.csv, camera_distance_recall.json   Ianna, YOLO11s CAM_FRONT, 2-D AP50, recall by distance
  ../results/mini_results.csv        LiDAR-only sweep, bootstrap CSV (shared schema)
  ../results/lane_recall_mini.csv    LiDAR-only recall on the camera lane's ground truth (common/tools/lane_recall.py)
  ../results/fullval_sweep_partial.csv  full-val LiDAR-only runs finished so far

    python lidar/report/make_comparison.py        # writes figures/*.png and tables.md next to this file
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
IN, RES, FIG = HERE / "inputs", HERE.parent / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)
C_L, C_F, C_FN, C_C = "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"


# ---------------------------------------------------------------- loaders
def lidar_mini():
    d = defaultdict(list)
    for r in csv.DictReader(open(RES / "mini_results.csv")):
        if r["metric"] in ("map", "nds") and r["value"] != "":
            d[(r["metric"], r["corruption"], int(r["severity"]))].append(float(r["value"]))
    return {k: sum(v) / len(v) for k, v in d.items()}


def fusion():
    return {(r["group"], r["condition"]): (float(r["mAP"]), float(r["NDS"]))
            for r in csv.DictReader(open(IN / "fusion_summary.csv"))}


def camera():
    return {(r["corruption"], int(r["severity"])): float(r["map2d_50"])
            for r in csv.DictReader(open(IN / "camera_accuracy_latency_summary.csv"))}


def lane_recall():
    d = defaultdict(list)
    for r in csv.DictReader(open(RES / "lane_recall_mini.csv")):
        d[(r["corruption"], int(r["severity"]), r["bin"])].append((float(r["recall"]), int(r["n_gt"])))
    return {k: sum(x for x, _ in v) / len(v) for k, v in d.items()}, {k: v[0][1] for k, v in d.items()}


def fullval():
    d = defaultdict(list)
    for r in csv.DictReader(open(RES / "fullval_sweep_partial.csv")):
        run = r["run"]
        if run == "clean":
            d[("clean", 0)].append((float(r["map"]), float(r["nds"])))
        else:
            name, sev, _ = run.rsplit("_", 2)
            d[(name, int(sev[1:]))].append((float(r["map"]), float(r["nds"])))
    return {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v), len(v)) for k, v in d.items()}


L, F, CAM = lidar_mini(), fusion(), camera()
LR, LRN = lane_recall()
FV = fullval()
l_clean = L[("map", "clean", 0)]
f_clean = F[("baseline", "clean")][0]
out = []   # markdown fragments


# ---------------------------------------------------------------- figure 1: LiDAR degradations, LiDAR-only vs fusion
def fig_lidar():
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    # point drop: x = fraction of points dropped
    lx = [0.10, 0.25, 0.40, 0.60, 0.80]
    ax[0].plot([0] + lx, [l_clean] + [L[("map", "point_drop", s)] for s in range(1, 6)], "o-", c=C_L, label="LiDAR-only")
    fx = [0.25, 0.5, 0.75, 0.9]
    ax[0].plot([0] + fx, [f_clean] + [F[("lidar_point_drop", f"drop{x:g}")][0] for x in fx], "s-", c=C_F, label="Fusion C+L")
    ax[0].plot(fx, [F[("lidar_stress_no_cams", f"drop{x:g}")][0] for x in fx[2:]] and [None, None] +
               [F[("lidar_stress_no_cams", f"drop{x:g}")][0] for x in fx[2:]], "^--", c=C_FN, label="Fusion, cameras black")
    ax[0].set(title="point drop", xlabel="fraction of points dropped")
    # beams: x = beams kept of 32
    lb = [24, 16, 12, 8, 4]
    ax[1].plot([32] + lb, [l_clean] + [L[("map", "beam_reduction", s)] for s in range(1, 6)], "o-", c=C_L)
    fb = [16, 4, 1]
    ax[1].plot([32] + fb, [f_clean] + [F[("lidar_beams", f"beams{b}")][0] for b in fb], "s-", c=C_F)
    ax[1].plot(fb, [F[("lidar_stress_no_cams", f"beams{b}")][0] for b in fb], "^--", c=C_FN)
    ax[1].set(title="beam reduction", xlabel="beams kept (of 32)"); ax[1].invert_xaxis()
    # range
    lr = [50, 40, 30, 20, 10]
    ax[2].plot([100] + lr, [l_clean] + [L[("map", "range_limit", s)] for s in range(1, 6)], "o-", c=C_L)
    fr = [40, 30, 20, 10]
    ax[2].plot([100] + fr, [f_clean] + [F[("lidar_range", f"range{r}m")][0] for r in fr], "s-", c=C_F)
    ax[2].plot([30, 20], [F[("lidar_stress_no_cams", "range30m")][0], F[("lidar_stress_no_cams", "range20m")][0]], "^--", c=C_FN)
    ax[2].set(title="range limit", xlabel="max range (m); 100 = none"); ax[2].invert_xaxis()
    # sweeps: x = past sweeps kept (of 9); LiDAR-only keeps 9*(1-frac) on average
    lk = [9 * (1 - f) for f in (0.2, 0.4, 0.6, 0.8, 1.0)]
    ax[3].plot([9] + lk, [l_clean] + [L[("map", "missing_sweeps", s)] for s in range(1, 6)], "o-", c=C_L)
    fk = [4, 1, 0]
    ax[3].plot([9] + fk, [f_clean] + [F[("lidar_sweeps", f"sweeps{k}")][0] for k in fk], "s-", c=C_F)
    ax[3].plot([0], [F[("lidar_stress_no_cams", "sweeps0")][0]], "^", c=C_FN)
    ax[3].set(title="missing sweeps", xlabel="past sweeps kept (of 9)"); ax[3].invert_xaxis()
    for a in ax:
        a.set_ylim(0, 0.65); a.grid(alpha=.3); a.set_ylabel("3-D mAP (mini-val, 81 samples)")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig1_lidar_degradations.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- figure 2: camera degradations
def fig_camera():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    c_clean = CAM[("clean", 0)]
    br = [(1.0, 0), (0.5, 1), (0.25, 2), (0.1, 3)]
    ax[0].plot([b for b, _ in br], [c_clean if s == 0 else CAM[("brightness", s)] for _, s in br], "o-", c=C_C, label="Camera YOLO11s (2-D AP50)")
    ax[0].plot([1.0, 0.5, 0.25, 0.1], [f_clean] + [F[("cam_brightness", f"bright{b:g}")][0] for b in (0.5, 0.25, 0.1)], "s-", c=C_F, label="Fusion C+L (3-D mAP)")
    ax[0].plot([1.0, 0.1], [l_clean, l_clean], "-", c=C_L, label="LiDAR-only (3-D mAP, unaffected)")
    ax[0].set(title="darkening (brightness scale)", xlabel="brightness scale"); ax[0].invert_xaxis()
    bl = [(0, 0), (2, 1), (5, 2), (10, 3)]
    ax[1].plot([b for b, _ in bl], [c_clean if s == 0 else CAM[("blur", s)] for _, s in bl], "o-", c=C_C)
    ax[1].plot([0, 2, 5, 10], [f_clean] + [F[("cam_blur", f"blur{b}px")][0] for b in (2, 5, 10)], "s-", c=C_F)
    ax[1].plot([0, 10], [l_clean, l_clean], "-", c=C_L)
    ax[1].set(title="Gaussian blur", xlabel="radius / sigma (px)")
    for a in ax:
        a.set_ylim(0, 0.65); a.grid(alpha=.3); a.set_ylabel("mAP (mini-val)")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig2_camera_degradations.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- figure 3: recall by distance
def fig_recall():
    cam = json.load(open(IN / "camera_distance_recall.json"))
    bins = [("near_0_20m", "0-20 m"), ("medium_20_40m", "20-40 m"), ("far_40m_plus", "40+ m")]
    series = [
        ("Camera clean", [cam["clean_0"][b]["recall"] for b, _ in bins], C_C, ""),
        ("Camera brightness 0.1", [cam["brightness_3"][b]["recall"] for b, _ in bins], C_C, "//"),
        ("Camera blur 10 px", [cam["blur_3"][b]["recall"] for b, _ in bins], C_C, ".."),
        ("LiDAR clean", [LR[("clean", 0, b)] for b, _ in bins], C_L, ""),
        ("LiDAR range 30 m", [LR[("range_limit", 3, b)] for b, _ in bins], C_L, "//"),
        ("LiDAR 4 beams", [LR[("beam_reduction", 5, b)] for b, _ in bins], C_L, ".."),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.13
    for i, (name, vals, col, hatch) in enumerate(series):
        ax.bar([j + (i - 2.5) * w for j in range(3)], vals, w, label=name, color=col, hatch=hatch, alpha=0.85 if not hatch else 0.55)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in bins]); ax.set_ylabel("recall of the same 1,310 camera-visible objects")
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "fig3_recall_by_distance.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------- figure 4: mini vs full-val consistency
def fig_fullval():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    sev = [1, 2, 3, 4]
    fv_clean = FV[("clean", 0)][0]
    ax.plot([0] + sev, [1.0] + [FV[("point_drop", s)][0] / fv_clean for s in sev], "o-", c=C_L, label="full val (6,019 samples), 3 seeds")
    ax.plot([0] + sev, [1.0] + [L[("map", "point_drop", s)] / l_clean for s in sev], "s--", c="gray", label="mini-val (81 samples), 3 seeds")
    ax.set(xlabel="point_drop severity", ylabel="mAP retention (RR)", title="LiDAR-only point drop: mini vs full val")
    ax.set_ylim(0.7, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig4_mini_vs_fullval.png", dpi=130); plt.close(fig)


fig_lidar(); fig_camera(); fig_recall(); fig_fullval()

# ---------------------------------------------------------------- tables
def rr(x, c):
    return f"{x:.3f} ({x / c:.0%})"


T = []
T.append("### Table A - LiDAR-side degradations, nearest comparable conditions (3-D mAP, mini-val; retention in brackets)\n")
T.append("| Condition | LiDAR-only | Fusion C+L | Fusion, cameras black |\n|---|---|---|---|")
rows = [
    ("clean", l_clean, f_clean, None),
    ("point drop 25 %", L[("map", "point_drop", 2)], F[("lidar_point_drop", "drop0.25")][0], None),
    ("point drop 75 % / 80 %*", L[("map", "point_drop", 5)], F[("lidar_point_drop", "drop0.75")][0], F[("lidar_stress_no_cams", "drop0.75")][0]),
    ("16 beams of 32", L[("map", "beam_reduction", 2)], F[("lidar_beams", "beams16")][0], F[("lidar_stress_no_cams", "beams16")][0]),
    ("4 beams of 32", L[("map", "beam_reduction", 5)], F[("lidar_beams", "beams4")][0], F[("lidar_stress_no_cams", "beams4")][0]),
    ("range 40 m", L[("map", "range_limit", 2)], F[("lidar_range", "range40m")][0], None),
    ("range 30 m", L[("map", "range_limit", 3)], F[("lidar_range", "range30m")][0], F[("lidar_stress_no_cams", "range30m")][0]),
    ("range 20 m", L[("map", "range_limit", 4)], F[("lidar_range", "range20m")][0], F[("lidar_stress_no_cams", "range20m")][0]),
    ("range 10 m", L[("map", "range_limit", 5)], F[("lidar_range", "range10m")][0], None),
    ("no past sweeps", L[("map", "missing_sweeps", 5)], F[("lidar_sweeps", "sweeps0")][0], F[("lidar_stress_no_cams", "sweeps0")][0]),
]
for name, a, b, c in rows:
    if name == "clean":
        T.append(f"| {name} | {a:.3f} | {b:.3f} | - |")
    else:
        T.append(f"| {name} | {rr(a, l_clean)} | {rr(b, f_clean)} | {rr(c, f_clean) if c is not None else '-'} |")
T.append("\n\\* LiDAR-only point drop is 80 % (severity 5) versus fusion's 75 %: not an exact match.\n")

T.append("### Table B - camera-side degradations (identical parameters in both lanes)\n")
T.append("| Condition | Camera YOLO11s (2-D AP50) | Fusion C+L (3-D mAP) | LiDAR-only (3-D mAP) |\n|---|---|---|---|")
c_clean = CAM[("clean", 0)]
T.append(f"| clean | {c_clean:.3f} | {f_clean:.3f} | {l_clean:.3f} |")
for lab, cs, fk in [("brightness x0.5", ("brightness", 1), ("cam_brightness", "bright0.5")),
                    ("brightness x0.25", ("brightness", 2), ("cam_brightness", "bright0.25")),
                    ("brightness x0.1", ("brightness", 3), ("cam_brightness", "bright0.1")),
                    ("blur 2 px", ("blur", 1), ("cam_blur", "blur2px")),
                    ("blur 5 px", ("blur", 2), ("cam_blur", "blur5px")),
                    ("blur 10 px", ("blur", 3), ("cam_blur", "blur10px"))]:
    T.append(f"| {lab} | {rr(CAM[cs], c_clean)} | {rr(F[fk][0], f_clean)} | {l_clean:.3f} (100 %, not affected) |")
T.append(f"| all cameras black | - | {rr(F[('cam_drop', 'all')][0], f_clean)} | {l_clean:.3f} |")

T.append("\n### Table C - recall of the camera lane's 1,310 objects, by distance (camera: 2-D IoU>=0.5, conf>=0.25; LiDAR: 3-D centre < 2 m, score>=0.25)\n")
T.append("| Condition | 0-20 m | 20-40 m | 40+ m | of which 40-54 m | of which >54 m |\n|---|---|---|---|---|---|")
cam = json.load(open(IN / "camera_distance_recall.json"))
n = {b: LRN[("clean", 0, b)] for b in ("near_0_20m", "medium_20_40m", "far_40m_plus", "far_40_54m", "far_54m_plus")}
T.append(f"| objects (n) | {n['near_0_20m']} | {n['medium_20_40m']} | {n['far_40m_plus']} | {n['far_40_54m']} | {n['far_54m_plus']} |")
for lab, key in [("Camera clean", "clean_0"), ("Camera brightness x0.1", "brightness_3"), ("Camera blur 10 px", "blur_3")]:
    T.append(f"| {lab} | {cam[key]['near_0_20m']['recall']:.3f} | {cam[key]['medium_20_40m']['recall']:.3f} | {cam[key]['far_40m_plus']['recall']:.3f} | - | - |")
for lab, c, s in [("LiDAR clean", "clean", 0), ("LiDAR range 30 m", "range_limit", 3), ("LiDAR range 20 m", "range_limit", 4),
                  ("LiDAR 16 beams", "beam_reduction", 2), ("LiDAR 4 beams", "beam_reduction", 5),
                  ("LiDAR point drop 80 %", "point_drop", 5), ("LiDAR no past sweeps", "missing_sweeps", 5)]:
    T.append(f"| {lab} | " + " | ".join(f"{LR[(c, s, b)]:.3f}" for b in ("near_0_20m", "medium_20_40m", "far_40m_plus", "far_40_54m", "far_54m_plus")) + " |")

T.append("\n### Table D - LiDAR-only on the full val set (6,019 samples), runs finished so far\n")
T.append("| Condition | mAP | NDS | RR (mAP) | same condition on mini-val RR |\n|---|---|---|---|---|")
fv_clean = FV[("clean", 0)][0]
T.append(f"| clean (deterministic sweeps) | {fv_clean:.4f} | {FV[('clean', 0)][1]:.4f} | 100 % | 100 % |")
for c in ("point_drop", "range_limit"):
    for s in range(1, 6):
        if (c, s) in FV:
            m, nd, k = FV[(c, s)]
            T.append(f"| {c} s{s} ({k} seed{'s' if k > 1 else ''}) | {m:.4f} | {nd:.4f} | {m / fv_clean:.1%} | {L[('map', c, s)] / l_clean:.1%} |")
(HERE / "tables.md").write_text("\n".join(T) + "\n", encoding="utf-8")
print("\n".join(T))
