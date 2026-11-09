import numpy as np
from sdc_camera.corruptions import apply
from sdc_camera.projection import common_class
from sdc_camera.evaluation_2d import evaluate,iou

def box(cls='car',token='a',confidence=0.9,x1=0,y1=0,x2=10,y2=10):
    return dict(cls=cls,sample_token=token,camera='CAM_FRONT',confidence=confidence,x1=x1,y1=y1,x2=x2,y2=y2)

def test_corruptions():
    a=np.full((40,40,3),200,dtype=np.uint8)
    assert np.array_equal(apply(a),a)
    assert apply(a,'brightness',2).mean()==50
    assert apply(a,'camera_drop',1).sum()==0
    assert apply(a,'blur',3).shape==a.shape

def test_mapping():
    assert common_class('vehicle.bus.rigid')=='bus'
    assert common_class('human.pedestrian.adult')=='person'
    assert common_class('vehicle.construction') is None

def test_ap_duplicate_and_false_positive():
    gt=[box()];pred=[box(confidence=.9),box(confidence=.8),box(confidence=.7,x1=20,x2=30)]
    out=evaluate(pred,gt,classes=['car'])
    assert out['per_class']['car']['ap2d_50']==1.
    assert out['per_class']['car']['tp']==1
    assert out['per_class']['car']['fp']==2

def test_ap_false_positive_before_tp():
    gt=[box()];pred=[box(confidence=.9,x1=20,x2=30),box(confidence=.8)]
    assert evaluate(pred,gt,classes=['car'])['map2d_50']==.5

def test_empty_gt_excluded():
    out=evaluate([],[],classes=['car'])
    assert out['map2d_50'] is None
