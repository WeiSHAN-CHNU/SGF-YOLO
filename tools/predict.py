import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\lijinfeng\ultralytics-11.20\runs\all\weights\best.pt')
    results = model.predict(source=r"D:\lijinfeng\ultralytics-11.20\ultralytics-main\dateset\shuju1\images\val",
                            project='D:/lijinfeng/ultralytics-11.20/predict/llvip',
                            name='ALL',
                batch=16,
                workers=0,
                device='0',
                show=False, save=True)

