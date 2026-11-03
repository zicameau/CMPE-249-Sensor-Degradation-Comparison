"""Generate LaTeX table snippets (report/tab_*.tex) directly from results/*.json (no hand-copied numbers)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
R = HERE / "results"
OUT = HERE / "report"
L = {"yolo11s": "YOLO11s", "rtdetr-l": "RT-DETR-L", "frcnn": "Faster R-CNN", "fcos": "FCOS",
     "yolo11n": "YOLO11n", "yolo11m": "YOLO11m"}
MAIN = ["yolo11s", "rtdetr-l", "frcnn", "fcos"]


def J(n):
    return json.loads((R / n).read_text())


def p(x, d=1):
    return f"{100 * x:.{d}f}"


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n")


bench = J("latency_bench.json")
extra = J("extra.json")
fail = J("failure.json")

# ---------- main results
t = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
     r"Model & Eval set & mAP & AP$_{50}$ & AP$_{75}$ & AP$_S$ & AP$_M$ & AP$_L$ & AR$_{100}$ \\", r"\midrule"]
for m in MAIN:
    c = J(f"{m}_coco.json"); k = J(f"{m}_kitti.json")
    for key, ds, nm in ((c, "metrics_coco80", "COCO val, 80 cls"), (c, "metrics_3cls", "COCO val, 3 cls"), (k, "metrics_3cls", "KITTI, 3 cls")):
        x = key[ds]
        t.append(f"{L[m] if nm.startswith('COCO val, 80') else ''} & {nm} & {p(x['mAP'])} & {p(x['AP50'])} & {p(x['AP75'])} & "
                 f"{p(x['APs'])} & {p(x['APm'])} & {p(x['APl'])} & {p(x['AR100'])} \\\\")
    t.append(r"\midrule" if m != MAIN[-1] else r"\bottomrule")
t.append(r"\end{tabular}")
write("tab_main.tex", t)

# ---------- per-class COCO vs KITTI
t = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
     r" & & \multicolumn{3}{c}{AP@[.5:.95]} & \multicolumn{3}{c}{AP$_{50}$} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
     r"Model & Set & person & car & truck & person & car & truck \\", r"\midrule"]
for m in MAIN:
    for ds, nm in (("coco", "COCO"), ("kitti", "KITTI")):
        pc = J(f"{m}_{ds}.json")["metrics_3cls"]["per_class"]
        t.append(f"{L[m] if ds == 'coco' else ''} & {nm} & " + " & ".join(p(pc[c]["AP"]) for c in ("person", "car", "truck")) + " & " +
                 " & ".join(p(pc[c]["AP50"]) for c in ("person", "car", "truck")) + r" \\")
    t.append(r"\midrule" if m != MAIN[-1] else r"\bottomrule")
t.append(r"\end{tabular}")
write("tab_perclass.tex", t)

# ---------- latency
t = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
     r"Model & Params (M) & \multicolumn{2}{c}{fp32 ms (KITTI / COCO)} & \multicolumn{2}{c}{fp16 ms (KITTI / COCO)} & \multicolumn{2}{c}{Peak mem. MB (fp32, KITTI / COCO)} & FPS$_{\rm fp32}$ COCO \\",
     r"\midrule"]
for m in ["yolo11n", "yolo11s", "yolo11m", "rtdetr-l", "frcnn", "fcos"]:
    par = J(f"{m}_kitti.json")["params_M"]
    g = lambda ds, pr: bench[f"{m}|{ds}|{pr}"]
    t.append(f"{L[m]} & {par:.1f} & {g('kitti','fp32')['median_ms']:.0f} & {g('coco','fp32')['median_ms']:.0f} & "
             f"{g('kitti','fp16')['median_ms']:.0f} & {g('coco','fp16')['median_ms']:.0f} & "
             f"{g('kitti','fp32')['peak_mem_MB']:.0f} & {g('coco','fp32')['peak_mem_MB']:.0f} & {g('coco','fp32')['fps']:.1f} \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
write("tab_latency.tex", t)

# ---------- ablations (KITTI)
rows = [("yolo11s_kitti", "YOLO11s, 640, NMS IoU 0.7 (default)"), ("yolo11s_nms0.3_kitti", "YOLO11s, 640, NMS IoU 0.3"),
        ("yolo11s_nms0.5_kitti", "YOLO11s, 640, NMS IoU 0.5"), ("yolo11s_nms0.9_kitti", "YOLO11s, 640, NMS IoU 0.9"),
        ("yolo11s_imgsz1280_kitti", "YOLO11s, 1280"), ("yolo11n_kitti", "YOLO11n, 640"), ("yolo11m_kitti", "YOLO11m, 640"),
        ("rtdetr-l_kitti", "RT-DETR-L, 640 (no NMS)"), ("rtdetr-l_imgsz1280_kitti", "RT-DETR-L, 1280 (no NMS)"),
        ("frcnn_kitti", "Faster R-CNN, short side 800 (default)"), ("fcos_kitti", "FCOS, short side 800 (default)")]
t = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Configuration (KITTI, 3 cls) & mAP & AP$_{50}$ & AP$_S$ & AP$_M$ & AP$_L$ & AR$_{100}$ \\", r"\midrule"]
for tag, nm in rows:
    x = J(tag + ".json")["metrics_3cls"]
    t.append(f"{nm} & {p(x['mAP'])} & {p(x['AP50'])} & {p(x['APs'])} & {p(x['APm'])} & {p(x['APl'])} & {p(x['AR100'])} \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
write("tab_ablation.tex", t)

# ---------- dataset stats
s = extra["stats"]
t = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Set & Class & \#GT & median box h (px) & \% small/$h{<}40$ & \% occluded & \% truncated \\", r"\midrule"]
for c in ("person", "car", "truck"):
    x = s["coco"][c]
    t.append(f"COCO val & {c} & {x['n']} & {x['median_h_px']:.0f} & {x['pct_small']:.0f} (area$<32^2$) & -- & -- \\\\")
t.append(r"\midrule")
for c in ("person", "car", "truck"):
    x = s["kitti"][c]
    t.append(f"KITTI subset & {c} & {x['n']} & {x['median_h_px']:.0f} & {x['pct_h_lt40']:.0f} & {x['pct_occluded']:.0f} & {x['pct_truncated']:.0f} \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
write("tab_stats.tex", t)

# ---------- failure analysis
def rec(tag, k):
    h = fail[tag]["recall"]
    return 100 * h[k][0] / max(1, h[k][1])


t = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
     r" & \multicolumn{4}{c}{by GT box height} & \multicolumn{3}{c}{by class} & \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
     r"Model & $<$25 & 25--40 & 40--80 & $>$80 & person & car & truck & all \\", r"\midrule"]
for m in MAIN:
    t.append(f"{L[m]} & " + " & ".join(f"{rec(m, 'all|' + b):.1f}" for b in ("<25px", "25-40px", "40-80px", ">80px")) + " & " +
             " & ".join(f"{rec(m, c + '|all'):.1f}" for c in ("person", "car", "truck")) + f" & {rec(m, 'all|all'):.1f} \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
write("tab_recall.tex", t)

causes = ["correct but score < thr", "wrong class (IoU>=0.5)", "poor localisation (0.1<=IoU<0.5)", "no detection near GT"]
t = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Model & low score & wrong class & poor loc. & nothing near & total missed \\", r"\midrule"]
for m in MAIN:
    c = fail[m]["causes"]
    t.append(f"{L[m]} & " + " & ".join(str(c.get(k, 0)) for k in causes) + f" & {sum(c.values())} \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
write("tab_causes.tex", t)

t = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Model & dets$\geq$0.3 & TP & same-class dup. (\%) & cross-class dup. pairs & FP person / car / truck \\", r"\midrule"]
for m in MAIN:
    d = extra["dups"][m]; f = fail[m]["fp"]
    t.append(f"{L[m]} & {d['dets']} & {d['tp']} & {d['dup_rate_pct']:.1f} & {d['cross_class_dup_pairs']} & "
             f"{f.get('person', 0)} / {f.get('car', 0)} / {f.get('truck', 0)} \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
write("tab_dups.tex", t)

# breakdown + batch
bd = J("latency_breakdown.json"); bt = J("latency_batch.json")
t = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Model (KITTI, fp32) & pre & network & post & wall & net.\ @bs1 & net.\ per img @bs16 \\", r"\midrule"]
for m, w in (("yolo11n", "yolo11n.pt"), ("yolo11s", "yolo11s.pt"), ("yolo11m", "yolo11m.pt"), ("rtdetr-l", "rtdetr-l.pt")):
    b = bd[f"{m}|kitti"]
    t.append(f"{L[m]} & {b['preprocess']:.1f} & {b['inference']:.1f} & {b['postprocess']:.1f} & {b['wall']:.1f} & "
             f"{bt[w + '|1']['per_img_ms']:.1f} & {bt[w + '|16']['per_img_ms']:.1f} \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
write("tab_breakdown.tex", t)
print("ok")
