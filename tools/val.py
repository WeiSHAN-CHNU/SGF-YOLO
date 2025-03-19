import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\lijinfeng\ultralytics-11.20\runs\v10\weights\best.pt')
    model.val(data=r'D:\lijinfeng\ultralytics-11.20\ultralytics-main\dateset\shuju1\mydata.yaml',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='buchong/val',
              name='exp',
              )