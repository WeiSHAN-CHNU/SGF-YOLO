# coding=gbk
from ultralytics import YOLO
# ����ѵ���õ�ģ�ͻ�������ṹ�����ļ�
model = YOLO('D:/86199/Downloads/chengxu/ultralytics-main/ultralytics-main/runs/detect/train6/weights/best.pt')
# model = YOLO('ultralytics/cfg/models/v8/yolov8n.yaml')
# ��ӡģ�Ͳ�����Ϣ

print(model.info(detailed=True))