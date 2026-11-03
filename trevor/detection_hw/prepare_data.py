"""Download and convert datasets to COCO-format ground truth.

COCO  : val2017 (5000 images) + instances_val2017.json (official).
KITTI : random 1000-image subset (seed 0) of the 7481-image *training* split of the
        KITTI 2D object detection benchmark. Only the sampled PNGs are fetched from the
        12.5 GB official zip via HTTP range requests (remotezip).

KITTI -> COCO taxonomy (documented in the report):
  evaluated categories : person(1), car(3), truck(8)   [COCO category ids]
  Car, Van             -> car
  Truck                -> truck
  Pedestrian, Person_sitting -> person
  Cyclist              -> ignore region (iscrowd) for person AND bicycle
                          (COCO detectors output rider + bike separately; KITTI has one box)
  Tram, Misc           -> ignore region (iscrowd) for car, truck
  DontCare             -> ignore region (iscrowd) for person, car, truck
Ignore regions are encoded as COCO iscrowd=1 boxes: detections that overlap them
(IoA >= IoU threshold) are neither TP nor FP under pycocotools.
"""
import argparse
import json
import random
import zipfile
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).parent / "data"
COCO_IMG_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
KITTI_IMG_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
KITTI_LBL_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"

PERSON, CAR, TRUCK, BICYCLE = 1, 3, 8, 2
EVAL_CATS = [(PERSON, "person"), (CAR, "car"), (TRUCK, "truck")]
KITTI_MAP = {  # kitti class -> list of (coco_cat, iscrowd)
    "Car": [(CAR, 0)],
    "Van": [(CAR, 0)],
    "Truck": [(TRUCK, 0)],
    "Pedestrian": [(PERSON, 0)],
    "Person_sitting": [(PERSON, 0)],
    "Cyclist": [(PERSON, 1), (BICYCLE, 1)],
    "Tram": [(CAR, 1), (TRUCK, 1)],
    "Misc": [(CAR, 1), (TRUCK, 1)],
    "DontCare": [(PERSON, 1), (CAR, 1), (TRUCK, 1)],
}


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print("downloading", url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def prep_coco():
    d = ROOT / "coco"
    if not (d / "val2017").exists():
        download(COCO_IMG_URL, d / "val2017.zip")
        with zipfile.ZipFile(d / "val2017.zip") as z:
            z.extractall(d)
    if not (d / "annotations" / "instances_val2017.json").exists():
        download(COCO_ANN_URL, d / "annotations.zip")
        with zipfile.ZipFile(d / "annotations.zip") as z:
            z.extract("annotations/instances_val2017.json", d)
    print("coco ready:", len(list((d / "val2017").glob("*.jpg"))), "images")


def prep_kitti(n=1000, seed=0):
    from remotezip import RemoteZip

    d = ROOT / "kitti"
    img_dir = d / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    download(KITTI_LBL_URL, d / "labels.zip")
    with zipfile.ZipFile(d / "labels.zip") as z:
        names = sorted(n_ for n_ in z.namelist() if n_.startswith("training/label_2/") and n_.endswith(".txt"))
        ids = [Path(n_).stem for n_ in names]
        rng = random.Random(seed)
        chosen = sorted(rng.sample(ids, n))
        labels = {i: z.read(f"training/label_2/{i}.txt").decode() for i in chosen}
    with RemoteZip(KITTI_IMG_URL) as rz:
        for i in chosen:
            out = img_dir / f"{i}.png"
            if not out.exists():
                out.write_bytes(rz.read(f"training/image_2/{i}.png"))
    images, anns = [], []
    aid = 1
    stats = {}
    for idx, i in enumerate(chosen, 1):
        w, h = Image.open(img_dir / f"{i}.png").size
        images.append({"id": idx, "file_name": f"{i}.png", "width": w, "height": h})
        for line in labels[i].splitlines():
            p = line.split()
            cls = p[0]
            x1, y1, x2, y2 = map(float, p[4:8])
            trunc, occl = float(p[1]), int(p[2])
            stats[cls] = stats.get(cls, 0) + 1
            for cat, crowd in KITTI_MAP.get(cls, []):
                anns.append({
                    "id": aid, "image_id": idx, "category_id": cat, "iscrowd": crowd,
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                    "kitti_class": cls, "truncated": trunc, "occluded": occl,
                })
                aid += 1
    gt = {"images": images, "annotations": anns,
          "categories": [{"id": c, "name": n_} for c, n_ in EVAL_CATS]}
    (d / "kitti_subset_gt.json").write_text(json.dumps(gt))
    print("kitti ready:", len(images), "images; raw KITTI class counts:", stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-n", type=int, default=1000)
    a = ap.parse_args()
    prep_coco()
    prep_kitti(a.kitti_n)
