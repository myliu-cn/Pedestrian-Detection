import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

class Logistic_Detector():
    def __init__(self, img_dir, model_dir):
        self.img_dir = img_dir
        self.model_dir = model_dir
        self.img = cv2.imread(self.img_dir)
        self.lr = load(self.model_dir)
        self.get_score(self.img)
        boxes_with_scores = self.generate_possible_boxes()
        boxes_with_scores = self.nms(boxes_with_scores)
        self.draw_boxes(boxes_with_scores)

    def generate_slidingwindow(self, stepsize=None, windowsizes=None):
        '''获得一张图片的滑动窗口和对于bounding box的位置'''
        height, width = self.img.shape[:2]
        if (stepsize==None):
            stepsize=(int(2*width//10), int(width//10))
        if (windowsizes==None):
                windowsizes=[(int(height/1.5), int(0.5*height/1.5)),
                                (int(2*width/3), int(width/3)), 
                                (int(2*width/2.5), int(width/2.5)),
                                (int(2*width/1.5), int(width/1.5))]
        images = []
        bounding_boxes = []
        for windowsize in windowsizes:
            for i in range(0, height-windowsize[0]+1, stepsize[0]):
                for j in range(0, width-windowsize[1]+1, stepsize[1]):
                    img_win = self.img[i:i+windowsize[0], j:j+windowsize[1]]
                    bounding_boxes.append((j,i,j+windowsize[1]-1,i+windowsize[0]-1))
                    win_resize = cv2.resize(img_win, (64,128), interpolation=cv2.INTER_LINEAR)
                    images.append(win_resize)
        return np.array(images), bounding_boxes

    def get_pedestrian_boxes(self, images):
        '''获取行人框的索引'''
        hog_features = []
        hog = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(32, 32), 
                        _blockStride=(16, 16), _cellSize=(16, 16), _nbins=9)
        for image in images:
            hog_feature = hog.compute(image)
            hog_feature = np.array(hog_feature, dtype=np.float32)
            hog_features.append(hog_feature)
        hog_features = np.array(hog_features, dtype=np.float32)
        result = self.lr.predict(hog_features)
        index = np.where(result==1)[0]
        return index
    
    def generate_possible_boxes(self):
        '''
        生成可能含有行人的框和对应的分数
        返回值：[(score, bounding_box), ...]        
        '''
        images, bounding_boxes = self.generate_slidingwindow()
        index = self.get_pedestrian_boxes(images)
        scores = []
        boxes_with_scores = []
        if index.shape[0] == 0:
            return []
        else:
            index = index.tolist()
            bounding_boxes = [bounding_boxes[i] for i in index]
            for bounding_box in bounding_boxes:
                x1, y1, x2, y2 = bounding_box
                img = self.img[y1:y2, x1:x2]
                scores.append(self.get_score(img))
            for i in range(len(bounding_boxes)):
                boxes_with_scores.append((scores[i], bounding_boxes[i]))
            return boxes_with_scores
        
    def get_score(self, img):
        hog = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(32, 32), 
                        _blockStride=(16, 16), _cellSize=(16, 16), _nbins=9)
        img = cv2.resize(img, (64,128), interpolation=cv2.INTER_LINEAR)
        hog_feature = hog.compute(img)
        hog_feature = np.array(hog_feature, dtype=np.float32)
        score = self.lr.predict_proba(hog_feature.reshape(1,-1))
        return score[0][1]
    
    def calculate_iou(self, rect1, rect2):
        x1, y1, x2, y2 = rect1
        x3, y3, x4, y4 = rect2
        if (x1 > x4 or x2 < x3 or y1 > y4 or y2 < y3):
            return 0
        else:
            x = min(x2, x4) - max(x1, x3)
            y = min(y2, y4) - max(y1, y3)
            return x*y/(abs(x1-x2)*abs(y1-y2)+abs(x3-x4)*abs(y3-y4)-x*y)
        
    def nms(self, boxes_with_scores, threshold = 0.15):
        '''
        >>> boxes_with_scores: [(score, bounding_box), ...]
        >>> threshold: 阈值
        Return: 非极大抑制后的 [(score, bounding_box), ...]
        '''
        if len(boxes_with_scores) == 0:
            return []
        boxes_with_scores.sort(key=lambda x:x[0], reverse=True)
        new_boxes_with_scores = []
        new_boxes_with_scores.append(boxes_with_scores[0])
        for i in range(1, len(boxes_with_scores)):
            flag = True
            for j in range(len(new_boxes_with_scores)):
                if self.calculate_iou(boxes_with_scores[i][1], new_boxes_with_scores[j][1]) > threshold:
                    flag = False
                    break
            if flag:
                new_boxes_with_scores.append(boxes_with_scores[i])
        return new_boxes_with_scores
    
    def draw_boxes(self, boxes_with_scores):
        '''
        >>> boxes_with_scores: [(score, bounding_box), ...]
        '''
        for box_with_score in boxes_with_scores:
            _, bounding_box = box_with_score
            x1, y1, x2, y2 = bounding_box
            cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 显示图片
        # cv2.imshow('img', self.img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 保存图片
        image_name = self.img_dir.split('/')[-1]
        pos_or_neg = self.img_dir.split('/')[-2]
        cv2.imwrite('/'.join(['./Logistic_Result', pos_or_neg, image_name]), self.img)


if __name__ == '__main__':
    # 读取测试图片
    with open('./INRIAPerson/Test/pos.lst', 'r') as f:
        pos_list = f.readlines()
    with open('./INRIAPerson/Test/neg.lst', 'r') as f:
        neg_list = f.readlines()

    # 测试正样本图片
    for pos in pos_list:
        pos = pos.strip()
        pos_img_dir = './INRIAPerson/' + pos
        Logistic_Detector(pos_img_dir, 'logistic_regression.joblib')

    # 测试负样本图片
    for neg in neg_list:
        neg = neg.strip()
        neg_img_dir = './INRIAPerson/' + neg
        Logistic_Detector(neg_img_dir, 'logistic_regression.joblib')

