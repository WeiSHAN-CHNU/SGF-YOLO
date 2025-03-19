import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(r'D:\lijinfeng\ultralytics-11.20\ultralytics-main\ultralytics\cfg\models\v8\yolov8-change.yaml')
  #      model = YOLO(r'D:\lijinfeng\ultralytics-11.20\runs\v35\weights\last.pt')
    model.train(data=r'D:\lijinfeng\ultralytics-11.20\ultralytics-main\dateset\shuju1\mydata.yaml',
                project='D:/lijinfeng/ultralytics-11.20/runs',
               name='SE+xiao+GS',
                epochs=200,
                batch=16,
                workers=0,
          device=0,
                )

# Train the model
# results = model.train(data='v7_data.yaml', epochs=150, imgsz=800)

