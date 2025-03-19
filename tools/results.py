# coding=gbk

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # �г�����ȡ�������ݵ��ļ�λ��
    # v5��v8����csv��ʽ�ģ�v7��txt��ʽ��
    result_dict = {
          'YOLOv8n': r'D:\lijinfeng\ultralytics-11.20\runs\train3\results.csv',
          'Ours': r'D:\lijinfeng\ultralytics-11.20\runs\all\results.csv',
    }

    # ����map50
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[6]).values.ravel()    # 6��ָmap50���±꣨ÿ�д�0��ʼ��������
        else:   # �ļ���׺��txt
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[10]))   # 10��ָmap50���±꣨ÿ�д�0��ʼ��������
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='1')   # ������ϸ��Ϊ1

    # ���x���y���ǩ
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.grid()
    # ��ʾͼ��
    plt.savefig("mAP50.png", dpi=600)   # dpi����Ϊ300/600/900����ʾ��Ϊ�������ʸ��ͼ
    plt.show()


    # ����map50-95
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[7]).values.ravel()    # 7��ָmap50-95���±꣨ÿ�д�0��ʼ��������
        else:
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[11]))   # 11��ָmap50-95���±꣨ÿ�д�0��ʼ��������
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='1')

    # ���x���y���ǩ
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5:0.95')
    plt.legend()
    plt.grid()
    # ��ʾͼ��
    plt.savefig("mAP50-95.png", dpi=600)
    plt.show()

    # ����ѵ������loss
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            box_loss = pd.read_csv(res_path, usecols=[1]).values.ravel()
            obj_loss = pd.read_csv(res_path, usecols=[2]).values.ravel()
            cls_loss = pd.read_csv(res_path, usecols=[3]).values.ravel()
            data = np.round(box_loss + obj_loss + cls_loss, 5)    # 3��loss��Ӳ��ұ���С�����5λ����v7һ�£�

        else:
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[5]))
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='1')

    # ���x���y���ǩ
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    # ��ʾͼ��
    plt.savefig("loss.png", dpi=600)
    plt.show()
