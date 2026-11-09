"""Dataset-level VOC-style continuous AP at IoU 0.50 (not COCO AP@[.50:.95])."""
from collections import defaultdict
import numpy as np
from .projection import COMMON_CLASSES

def iou(a,b):
    x1=max(a['x1'],b['x1']);y1=max(a['y1'],b['y1'])
    x2=min(a['x2'],b['x2']);y2=min(a['y2'],b['y2'])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a['x2']-a['x1'])*max(0,a['y2']-a['y1'])
    bb=max(0,b['x2']-b['x1'])*max(0,b['y2']-b['y1'])
    return intersection/(aa+bb-intersection) if aa+bb>intersection else 0.0

def evaluate(predictions, ground_truth, classes=COMMON_CLASSES, threshold=.5, fixed_conf=.25):
    """Rows: GT {sample_token,camera,cls,x1,y1,x2,y2}; pred same plus confidence.
    Only samples explicitly represented by ground_truth OR supplied in predictions
    are included; use a fixed sample manifest externally to include empty frames.
    AP for classes with no GT is None and excluded from macro mean.
    """
    out={};aps=[]
    for cls in classes:
        gt=[g for g in ground_truth if g['cls']==cls]
        ps=sorted((p for p in predictions if p['cls']==cls),key=lambda p:-float(p['confidence']))
        by_frame=defaultdict(list)
        for i,g in enumerate(gt):by_frame[(g['sample_token'],g.get('camera','CAM_FRONT'))].append(i)
        matched=set(); tp=[];fp=[]
        for p in ps:
            key=(p['sample_token'],p.get('camera','CAM_FRONT'))
            best=-1;score=0.0
            for idx in by_frame[key]:
                if idx in matched:continue
                v=iou(p,gt[idx])
                if v>score:score=v;best=idx
            hit=best>=0 and score>=threshold
            if hit:matched.add(best)
            tp.append(int(hit));fp.append(int(not hit))
        tp=np.asarray(tp,dtype=float);fp=np.asarray(fp,dtype=float)
        cumtp=np.cumsum(tp);cumfp=np.cumsum(fp)
        precision=cumtp/np.maximum(cumtp+cumfp,1)
        recall=cumtp/len(gt) if gt else np.zeros(len(ps))
        if gt:
            # precision envelope, exact step integration
            mrec=np.r_[0.,recall,1.];mpre=np.r_[0.,precision,0.]
            for i in range(len(mpre)-2,-1,-1):mpre[i]=max(mpre[i],mpre[i+1])
            inds=np.where(mrec[1:]!=mrec[:-1])[0]
            ap=float(np.sum((mrec[inds+1]-mrec[inds])*mpre[inds+1]))
            aps.append(ap)
        else:ap=None
        fixed=[i for i,p in enumerate(ps) if float(p['confidence'])>=fixed_conf]
        n_tp=int(tp[fixed].sum()) if fixed else 0
        n_fp=int(fp[fixed].sum()) if fixed else 0
        out[cls]={'ap2d_50':ap,'gt_count':len(gt),'tp':n_tp,'fp':n_fp,'fn':len(gt)-n_tp,
                  'precision':n_tp/(n_tp+n_fp) if n_tp+n_fp else 0.,
                  'recall':n_tp/len(gt) if gt else None}
    return {'per_class':out,'map2d_50':float(np.mean(aps)) if aps else None,
            'macro_class_count':len(aps)}
