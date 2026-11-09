"""nuScenes 3D annotations -> camera 2D bounding rectangles.
Conservative policy: omit boxes crossing the near plane. Verify visibility rules
with teammates before interpreting results as benchmark scores.
"""
import numpy as np
from PIL import Image

COMMON_CLASSES=('car','truck','bus','person','bicycle','motorcycle')

def common_class(name):
    if name in ('vehicle.car','vehicle.truck','vehicle.bicycle','vehicle.motorcycle'):
        return name.split('.')[-1]
    if name.startswith('vehicle.bus.') or name == 'vehicle.bus': return 'bus'
    if name.startswith('human.pedestrian.'): return 'person'
    return None

def project_sample(nusc, sample, camera='CAM_FRONT'):
    from nuscenes.utils.geometry_utils import view_points
    token=sample['data'][camera]
    path,boxes,K=nusc.get_sample_data(token)
    with Image.open(path) as im: width,height=im.size
    rows=[]
    for box in boxes:
        cls=common_class(box.name)
        if cls is None: continue
        corners=box.corners()
        if np.any(corners[2]<=0.1): continue
        uv=view_points(corners,np.asarray(K),normalize=True)[:2]
        x1,y1=uv.min(axis=1);x2,y2=uv.max(axis=1)
        x1,x2=np.clip([x1,x2],0,width);y1,y2=np.clip([y1,y2],0,height)
        if x2<=x1 or y2<=y1: continue
        rows.append(dict(sample_token=sample['token'],camera=camera,cls=cls,
                         x1=float(x1),y1=float(y1),x2=float(x2),y2=float(y2),
                         distance_m=float(np.linalg.norm(box.center)),
                         annotation_token=getattr(box,'token',None) or ''))
    return rows
