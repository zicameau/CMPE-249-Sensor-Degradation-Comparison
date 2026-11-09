"""Run: python scripts/run_camera_sweep.py --dataroot /path/to/nuscenes --out results/camera
Run from camera/ root, with `pip install -e .` first.
"""
import argparse,csv,json
from pathlib import Path
import cv2
from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO
from sdc_camera.detector_core import predict_image
from sdc_camera.projection import project_sample,common_class
from sdc_camera.evaluation_2d import evaluate

def save_csv(path,rows,columns):
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataroot',required=True);p.add_argument('--out',default='results/camera')
    p.add_argument('--version',default='v1.0-mini');p.add_argument('--model',default='yolo11s.pt')
    p.add_argument('--limit',type=int,default=20);p.add_argument('--manifest',help='Optional newline-delimited sample tokens')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    nusc=NuScenes(version=args.version,dataroot=args.dataroot,verbose=False)
    if args.manifest:
        tokens=[s.strip() for s in Path(args.manifest).read_text().splitlines() if s.strip()]
        samples=[nusc.get('sample',t) for t in tokens]
    else:samples=nusc.sample[:args.limit]
    (out/'sample_tokens.txt').write_text('\n'.join(s['token'] for s in samples)+'\n')
    model=YOLO(args.model);gt=[];pred=[]
    conditions=[('clean',0)]+[(kind,s) for kind in ('brightness','blur') for s in (1,2,3)]
    for sample in samples:
        gt.extend(project_sample(nusc,sample))
        camera=nusc.get('sample_data',sample['data']['CAM_FRONT'])
        image=cv2.imread(str(Path(args.dataroot)/camera['filename']))
        if image is None:raise FileNotFoundError(camera['filename'])
        for kind,s in conditions:
            for row in predict_image(model,image,sample['token'],corruption=kind,severity=s):
                cls=common_class('vehicle.'+row['class_name_coco']) if row['class_name_coco'] in ('car','truck','bus','bicycle','motorcycle') else ('person' if row['class_name_coco']=='person' else None)
                if cls:
                    row['cls']=cls;pred.append(row)
    save_csv(out/'ground_truth.csv',gt,['sample_token','camera','cls','x1','y1','x2','y2','distance_m','annotation_token'])
    save_csv(out/'predictions.csv',pred,['sample_token','camera','condition','severity','cls','confidence','x1','y1','x2','y2'])
    summaries={}
    for kind,s in conditions:
        subset=[r for r in pred if r['condition']==kind and r['severity']==s]
        summaries[f'{kind}_{s}']=evaluate(subset,gt)
    (out/'metrics.json').write_text(json.dumps(summaries,indent=2))
    print('Saved',out/'metrics.json')
if __name__=='__main__':main()
