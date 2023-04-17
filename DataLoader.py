import cv2
import random
import numpy as np


class DataLoader():
    def __init__(self):
        with open('./INRIAPerson/Train/pos.lst', 'r') as f:
            pos_list = f.readlines()
        with open('./INRIAPerson/Train/neg.lst', 'r') as f:
            neg_list = f.readlines()
        with open('./INRIAPerson/Train/annotations.lst', 'r') as f:
            annotation_list = f.readlines()
        images = []
        labels = []

        #读取所有正样本
        for n in range(len(pos_list)):
            pos = pos_list[n]
            pos = pos.strip() #去掉列表中每一个元素的换行符
            pos_img_dir = './INRIAPerson/' + pos
            annotation = annotation_list[n]
            annotation = annotation.strip()
            annotation_dir = './INRIAPerson/' + annotation
            pos_img = self.load_pos_img(pos_img_dir, annotation_dir)
            pos_label = [1]*len(pos_img)
            images = images + pos_img
            labels = labels + pos_label
        
        #读取所有负样本
        for neg in neg_list:
            neg = neg.strip()
            neg_img_dir = './INRIAPerson/' + neg
            neg_img = self.load_neg_img(neg_img_dir)
            neg_label = [0]*len(neg_img)
            images = images + neg_img
            labels = labels + neg_label
        
        #打乱数据集
        images = np.array(images)
        labels = np.array(labels)
        images, labels = self.shuffle(images, labels)
        self.images = images
        self.labels = labels
    
    def shuffle(self, img, label):
        #打乱数据集
        index = [i for i in range(len(img))]
        random.shuffle(index)
        img = img[index]
        label = label[index]
        return img, label

    def load_pos_img(self, img_dir, annotation_dir):
        #所有groundtruth_boxes位置
        groundtruth_boxes=[]
        #读取文件中所有groundtruth_boxes位置
        with open(annotation_dir,encoding='ANSI') as f:
            lines = f.readlines()
        for line in lines:
            if 'Bounding box for object' in line:
                coords = line.split(': ')[-1].strip().split(' - ')
                xmin, ymin = tuple(map(int, coords[0][1:-1].split(', ')))
                xmax, ymax = tuple(map(int, coords[1][1:-1].split(', ')))
                groundtruth_boxes.append((xmin, ymin, xmax, ymax))
        #读取图片并剪裁
        img = cv2.imread(img_dir)
        #返回值，groundtruth_boxes
        boxes=[]
        for box in groundtruth_boxes:
            img_win = img[box[1]:box[3], box[0]:box[2]]
            win_resize = cv2.resize(img_win, (64,128), interpolation=cv2.INTER_LINEAR)
            boxes.append(win_resize)
        return boxes

    def random_num(self):
        n = random.randint(0, 3)
        if n == 0:
            return 0
        else:
            return 1
        
    def load_neg_img(self, img_dir):
        #对于负样本，进行随机剪裁
        img = cv2.imread(img_dir)
        height = img.shape[0]  #输入图像的高度
        width = img.shape[1]   #输入图像的宽度
        boxes=[]
        win_resize = cv2.resize(img, (64,128), interpolation=cv2.INTER_LINEAR)
        boxes.append(win_resize)
        #随机剪裁
        for n in range(self.random_num()):
            x = random.randint(0, width//2)
            y = random.randint(0, height//2)
            img_win = img[y:y+height//3, x:x+width//3]
            win_resize = cv2.resize(img_win, (64,128), interpolation=cv2.INTER_LINEAR)
            boxes.append(win_resize)
        return boxes
    



    