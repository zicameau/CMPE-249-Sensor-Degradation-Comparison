### Table A - LiDAR-side degradations, nearest comparable conditions (3-D mAP, mini-val; retention in brackets)

| Condition | LiDAR-only | Fusion C+L | Fusion, cameras black |
|---|---|---|---|
| clean | 0.542 | 0.582 | - |
| point drop 25 % | 0.534 (99%) | 0.576 (99%) | - |
| point drop 75 % / 80 %* | 0.428 (79%) | 0.533 (92%) | 0.441 (76%) |
| 16 beams of 32 | 0.418 (77%) | 0.532 (91%) | 0.430 (74%) |
| 4 beams of 32 | 0.100 (18%) | 0.496 (85%) | 0.376 (65%) |
| range 40 m | 0.522 (96%) | 0.554 (95%) | - |
| range 30 m | 0.454 (84%) | 0.487 (84%) | 0.429 (74%) |
| range 20 m | 0.303 (56%) | 0.330 (57%) | 0.280 (48%) |
| range 10 m | 0.057 (10%) | 0.075 (13%) | - |
| no past sweeps | 0.402 (74%) | 0.505 (87%) | 0.371 (64%) |

\* LiDAR-only point drop is 80 % (severity 5) versus fusion's 75 %: not an exact match.

### Table B - camera-side degradations (identical parameters in both lanes)

| Condition | Camera YOLO11s (2-D AP50) | Fusion C+L (3-D mAP) | LiDAR-only (3-D mAP) |
|---|---|---|---|
| clean | 0.344 | 0.582 | 0.542 |
| brightness x0.5 | 0.369 (107%) | 0.569 (98%) | 0.542 (100 %, not affected) |
| brightness x0.25 | 0.339 (98%) | 0.545 (94%) | 0.542 (100 %, not affected) |
| brightness x0.1 | 0.203 (59%) | 0.522 (90%) | 0.542 (100 %, not affected) |
| blur 2 px | 0.322 (93%) | 0.571 (98%) | 0.542 (100 %, not affected) |
| blur 5 px | 0.232 (67%) | 0.547 (94%) | 0.542 (100 %, not affected) |
| blur 10 px | 0.129 (37%) | 0.523 (90%) | 0.542 (100 %, not affected) |
| all cameras black | - | 0.526 (90%) | 0.542 |

### Table C - recall of the camera lane's 1,310 objects, by distance (camera: 2-D IoU>=0.5, conf>=0.25; LiDAR: 3-D centre < 2 m, score>=0.25)

| Condition | 0-20 m | 20-40 m | 40+ m | of which 40-54 m | of which >54 m |
|---|---|---|---|---|---|
| objects (n) | 306 | 565 | 439 | 249 | 190 |
| Camera clean | 0.598 | 0.441 | 0.333 | - | - |
| Camera brightness x0.1 | 0.461 | 0.267 | 0.203 | - | - |
| Camera blur 10 px | 0.415 | 0.216 | 0.089 | - | - |
| LiDAR clean | 0.892 | 0.673 | 0.096 | 0.169 | 0.000 |
| LiDAR range 30 m | 0.892 | 0.451 | 0.000 | 0.000 | 0.000 |
| LiDAR range 20 m | 0.863 | 0.021 | 0.000 | 0.000 | 0.000 |
| LiDAR 16 beams | 0.824 | 0.444 | 0.018 | 0.032 | 0.000 |
| LiDAR 4 beams | 0.418 | 0.053 | 0.000 | 0.000 | 0.000 |
| LiDAR point drop 80 % | 0.780 | 0.317 | 0.019 | 0.033 | 0.000 |
| LiDAR no past sweeps | 0.807 | 0.476 | 0.032 | 0.056 | 0.000 |

### Table D - LiDAR-only on the full val set (6,019 samples), runs finished so far

| Condition | mAP | NDS | RR (mAP) | same condition on mini-val RR |
|---|---|---|---|---|
| clean (deterministic sweeps) | 0.6466 | 0.6921 | 100 % | 100 % |
| point_drop s1 (3 seeds) | 0.6415 | 0.6894 | 99.2% | 99.8% |
| point_drop s2 (3 seeds) | 0.6330 | 0.6843 | 97.9% | 98.6% |
| point_drop s3 (3 seeds) | 0.6186 | 0.6758 | 95.7% | 96.7% |
| point_drop s4 (3 seeds) | 0.5815 | 0.6529 | 89.9% | 91.9% |
| range_limit s3 (1 seed) | 0.5019 | 0.6236 | 77.6% | 83.7% |
