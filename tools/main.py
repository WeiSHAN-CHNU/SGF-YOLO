import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-obb.yaml')
    # model.load('yolov8n-obb.pt')
    model.train(data='dotav1.0.yaml',
                project='runs/obb/train/s/dotav1',
                name='v8',
                patience=100,
                epochs=150,
                batch=2,
                workers=8,
                imgsz=640,
                )

# Train the model
# results = model.train(data='v7_data.yaml', epochs=150, imgsz=800)