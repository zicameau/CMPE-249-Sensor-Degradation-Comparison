# Full experiment log -- every command used for the report.  Run from detection_hw/ :
#   .\.venv\Scripts\python prepare_data.py
#   powershell -File run_all.ps1
# stdout of each run goes to results/logs/<name>.log
$py = ".\.venv\Scripts\python"
New-Item -ItemType Directory -Force results\logs | Out-Null
function Run($name, $argv) {
    Write-Host "=== $name : evaluate.py $argv"
    & $py evaluate.py @argv *> "results\logs\$name.log"
}

# Main grid on KITTI (default thresholds: conf=0.001 for mAP, NMS IoU = model default)
foreach ($m in "yolo11s", "rtdetr-l", "frcnn", "fcos") { Run "${m}_kitti" @("--model", $m, "--dataset", "kitti") }

# Ablation 1: NMS-IoU sensitivity of YOLO11s on KITTI (RT-DETR has no NMS -> no counterpart)
foreach ($iou in 0.3, 0.5, 0.9) { Run "yolo11s_nms${iou}_kitti" @("--model", "yolo11s", "--dataset", "kitti", "--iou", $iou, "--tag", "nms$iou") }

# Ablation 2: input resolution on KITTI (1242x375 images; default long side 640)
foreach ($m in "yolo11s", "rtdetr-l") { Run "${m}_imgsz1280_kitti" @("--model", $m, "--dataset", "kitti", "--imgsz", 1280, "--tag", "imgsz1280") }

# Ablation 3: model size within the YOLO11 family (speed/accuracy frontier)
foreach ($m in "yolo11n", "yolo11m") { Run "${m}_kitti" @("--model", $m, "--dataset", "kitti") }

# Main grid on COCO val2017 (5000 images)
foreach ($m in "yolo11s", "rtdetr-l", "frcnn", "fcos") { Run "${m}_coco" @("--model", $m, "--dataset", "coco") }

& $py analyze.py

# Controlled latency/memory benchmarks and post-hoc analyses (order matters: analyze.py writes results/failure.json)
& $py bench_latency.py   *> results\logs\bench_latency.log
& $py bench_breakdown.py *> results\logs\bench_breakdown.log
& $py bench_batch.py     *> results\logs\bench_batch.log
& $py analyze.py
& $py extra_analysis.py
& $py taxonomy_check.py
& $py make_tables.py
# cd report; pdflatex report.tex; pdflatex report.tex
