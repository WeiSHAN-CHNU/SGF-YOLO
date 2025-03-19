import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('D:/lijinfeng/ultralytics-11.20/run_new/c2f/c2f-faster-EMA/weights/best.pt')
    model.val(data='D:/lijinfeng/ultralytics-11.20/ultralytics-main/dateset/visdrone2019/data.yaml',
                project='D:/lijinfeng/ultralytics-11.20/test/c2f',
               name='c2f-faster-EMA',
                batch=32,
                workers=0,
                )


