"""No ROS dependency; works in Colab and a local Python environment."""
from .corruptions import apply

def predict_image(model, image_bgr, sample_token, camera='CAM_FRONT', corruption='clean', severity=0, conf=0.001):
    modified = apply(image_bgr, corruption, severity)
    result = model.predict(modified, conf=conf, verbose=False)[0]
    rows=[]
    for box in result.boxes:
        cls=int(box.cls[0]); x1,y1,x2,y2=map(float,box.xyxy[0].tolist())
        rows.append(dict(sample_token=sample_token,camera=camera,condition=corruption,
                         severity=severity,class_name_coco=model.names[cls],
                         confidence=float(box.conf[0]),x1=x1,y1=y1,x2=x2,y2=y2))
    return rows
