import cv2
import numpy as np
from Detector import Detector

# 读取测试图片
with open('./INRIAPerson/Test/pos.lst', 'r') as f:
    pos_list = f.readlines()
with open('./INRIAPerson/Test/neg.lst', 'r') as f:
    neg_list = f.readlines()

# 测试正样本图片
for pos in pos_list:
    pos = pos.strip()
    pos_img_dir = './INRIAPerson/' + pos
    Detector(pos_img_dir, 'svm_model.xml')

# 测试负样本图片
for neg in neg_list:
    neg = neg.strip()
    neg_img_dir = './INRIAPerson/' + neg
    Detector(neg_img_dir, 'svm_model.xml')