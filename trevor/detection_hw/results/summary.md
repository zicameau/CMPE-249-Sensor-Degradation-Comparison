## Main results

| Model | Dataset | mAP | AP50 | AP75 | APs | APm | APl | AR100 | ms (median) | FPS | p95 ms | GPU MB | Params M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11s | COCO-80 | 45.5 | 62.0 | 49.2 | 27.6 | 50.1 | 62.5 | 60.2 | 37.7 | 26.6 | 60.6 | 111 | 9.5 |
| YOLO11s | COCO 3-cls | 46.3 | 65.7 | 49.9 | 27.7 | 55.6 | 69.7 | 60.6 | 37.7 | 26.6 | 60.6 | 111 | 9.5 |
| YOLO11s | KITTI 3-cls | 27.4 | 50.2 | 27.4 | 16.6 | 25.6 | 39.8 | 43.4 | 57.4 | 17.4 | 84.5 | 83 | 9.5 |
| RT-DETR-L | COCO-80 | 52.6 | 70.9 | 57.0 | 33.9 | 57.3 | 70.7 | 69.0 | 100.5 | 9.9 | 134.2 | 245 | 33.0 |
| RT-DETR-L | COCO 3-cls | 52.4 | 74.5 | 56.2 | 33.9 | 61.1 | 75.4 | 69.4 | 100.5 | 9.9 | 134.2 | 245 | 33.0 |
| RT-DETR-L | KITTI 3-cls | 32.0 | 55.6 | 31.7 | 22.1 | 31.6 | 44.0 | 56.1 | 112.0 | 8.9 | 148.5 | 245 | 33.0 |
| Faster R-CNN R50-FPN | COCO-80 | 37.2 | 59.0 | 39.9 | 21.3 | 40.5 | 48.4 | 53.0 | 197.2 | 5.1 | 220.3 | 664 | 41.8 |
| Faster R-CNN R50-FPN | COCO 3-cls | 41.8 | 68.8 | 43.8 | 28.6 | 49.0 | 56.2 | 58.4 | 197.2 | 5.1 | 220.3 | 664 | 41.8 |
| Faster R-CNN R50-FPN | KITTI 3-cls | 31.4 | 57.7 | 31.1 | 35.1 | 29.9 | 40.6 | 52.5 | 148.6 | 6.7 | 169.3 | 432 | 41.8 |
| FCOS R50-FPN | COCO-80 | 39.2 | 58.3 | 42.0 | 22.2 | 42.4 | 51.3 | 57.1 | 199.2 | 5.0 | 225.3 | 423 | 32.3 |
| FCOS R50-FPN | COCO 3-cls | 43.2 | 66.4 | 46.2 | 28.1 | 49.9 | 61.5 | 61.3 | 199.2 | 5.0 | 225.3 | 423 | 32.3 |
| FCOS R50-FPN | KITTI 3-cls | 33.6 | 59.0 | 35.9 | 28.1 | 31.7 | 43.3 | 52.5 | 140.5 | 7.1 | 168.5 | 283 | 32.3 |
| YOLO11n | KITTI 3-cls | 24.9 | 47.0 | 24.0 | 10.7 | 23.4 | 39.2 | 42.6 | 39.9 | 25.1 | 67.8 | 50 | 2.6 |
| YOLO11m | KITTI 3-cls | 30.4 | 54.5 | 30.4 | 22.4 | 28.6 | 43.0 | 48.2 | 45.5 | 22.0 | 74.4 | 137 | 20.1 |

## Per-class AP (3-class subset; AP / AP50)

| Model | Dataset | person | car | truck |
|---|---|---|---|---|
| YOLO11s | coco | 58.6 / 81.4 | 44.8 / 66.2 | 35.5 / 49.4 |
| YOLO11s | kitti | 27.4 / 58.1 | 45.0 / 78.4 | 9.6 / 14.1 |
| RT-DETR-L | coco | 61.5 / 85.9 | 49.9 / 73.4 | 45.8 / 64.1 |
| RT-DETR-L | kitti | 28.3 / 59.7 | 49.9 / 82.6 | 17.7 / 24.6 |
| Faster R-CNN R50-FPN | coco | 51.8 / 82.1 | 41.1 / 68.2 | 32.5 / 56.2 |
| Faster R-CNN R50-FPN | kitti | 29.2 / 61.7 | 44.8 / 79.9 | 20.2 / 31.4 |
| FCOS R50-FPN | coco | 53.3 / 80.6 | 42.2 / 65.2 | 34.2 / 53.4 |
| FCOS R50-FPN | kitti | 28.4 / 58.3 | 47.6 / 82.4 | 24.8 / 36.3 |

## Ablations on KITTI (3-class mAP / AP50 / AR100 / median ms)

| Config | mAP | AP50 | APs | APm | APl | AR100 | ms |
|---|---|---|---|---|---|---|---|
| YOLO11s imgsz640 NMS0.7 (default) | 27.4 | 50.2 | 16.6 | 25.6 | 39.8 | 43.4 | 57.4 |
| YOLO11s NMS IoU 0.3 | 26.4 | 48.1 | 16.5 | 24.2 | 38.6 | 40.2 | 46.7 |
| YOLO11s NMS IoU 0.5 | 26.8 | 49.3 | 16.6 | 24.8 | 39.2 | 41.5 | 44.9 |
| YOLO11s NMS IoU 0.9 | 26.9 | 48.0 | 15.6 | 24.8 | 40.3 | 45.3 | 46.4 |
| YOLO11s imgsz1280 | 31.2 | 55.6 | 19.1 | 31.7 | 40.4 | 48.2 | 54.8 |
| RT-DETR-L imgsz640 | 32.0 | 55.6 | 22.1 | 31.6 | 44.0 | 56.1 | 112.0 |
| RT-DETR-L imgsz1280 | 34.8 | 56.6 | 31.3 | 34.7 | 45.4 | 59.2 | 254.6 |

## KITTI failure analysis (operating threshold score >= 0.3, match IoU >= 0.5)

### Recall by GT box height (all classes)

| Model | <25px | 25-40px | 40-80px | >80px | overall |
|---|---|---|---|---|---|
| YOLO11s | 57.6 (n=571) | 66.6 (n=1269) | 74.2 (n=1610) | 82.6 (n=1526) | 73.0 |
| RT-DETR-L | 85.1 (n=571) | 79.6 (n=1269) | 80.8 (n=1610) | 85.5 (n=1526) | 82.4 |
| Faster R-CNN R50-FPN | 86.3 (n=571) | 82.5 (n=1269) | 81.5 (n=1610) | 85.7 (n=1526) | 83.6 |
| FCOS R50-FPN | 88.8 (n=571) | 85.6 (n=1269) | 85.1 (n=1610) | 87.9 (n=1526) | 86.5 |

### Recall by class

| Model | person | car | truck |
|---|---|---|---|
| YOLO11s | 59.2 (n=568) | 76.3 (n=4261) | 28.6 (n=147) |
| RT-DETR-L | 65.8 (n=568) | 85.3 (n=4261) | 61.9 (n=147) |
| Faster R-CNN R50-FPN | 70.2 (n=568) | 85.8 (n=4261) | 70.7 (n=147) |
| FCOS R50-FPN | 72.5 (n=568) | 88.6 (n=4261) | 80.3 (n=147) |

### Recall by KITTI occlusion level (0=visible, 1=partly, 2=largely, 3=unknown)

| Model | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| YOLO11s | 87.3 (n=2334) | 72.0 (n=1377) | 46.6 (n=1049) | 51.9 (n=216) |
| RT-DETR-L | 93.2 (n=2334) | 86.1 (n=1377) | 58.0 (n=1049) | 61.6 (n=216) |
| Faster R-CNN R50-FPN | 94.6 (n=2334) | 86.3 (n=1377) | 59.9 (n=1049) | 62.0 (n=216) |
| FCOS R50-FPN | 95.8 (n=2334) | 89.6 (n=1377) | 64.8 (n=1049) | 71.3 (n=216) |

### Why GT objects are missed (count of missed GT)

| Model | correct but score < thr | wrong class (IoU>=0.5) | poor localisation (0.1<=IoU<0.5) | no detection near GT | total missed |
|---|---|---|---|---|---|
| YOLO11s | 669 | 125 | 319 | 233 | 1346 |
| RT-DETR-L | 367 | 11 | 382 | 115 | 875 |
| Faster R-CNN R50-FPN | 406 | 18 | 285 | 107 | 816 |
| FCOS R50-FPN | 228 | 18 | 337 | 88 | 671 |

### False positives at threshold (not matching GT or ignore region)

| Model | person FP / dets | car FP / dets | truck FP / dets |
|---|---|---|---|
| YOLO11s | 130 / 615 | 599 / 4397 | 145 / 220 |
| RT-DETR-L | 350 / 1082 | 1448 / 6797 | 656 / 868 |
| Faster R-CNN R50-FPN | 1046 / 2158 | 2910 / 9158 | 440 / 653 |
| FCOS R50-FPN | 1998 / 3293 | 4281 / 10951 | 2137 / 2680 |