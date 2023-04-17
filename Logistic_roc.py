import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

def generate_slidingwindow(img, stepsize=None, windowsizes=None):
    '''获得一张图片的滑动窗口和对于bounding box的位置'''
    height, width = img.shape[:2]
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
                img_win = img[i:i+windowsize[0], j:j+windowsize[1]]
                bounding_boxes.append((j,i,j+windowsize[1]-1,i+windowsize[0]-1))
                win_resize = cv2.resize(img_win, (64,128), interpolation=cv2.INTER_LINEAR)
                images.append(win_resize)
    return np.array(images), bounding_boxes

def calculate_iou(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    if (x1 > x4 or x2 < x3 or y1 > y4 or y2 < y3):
        return 0
    else:
        x = min(x2, x4) - max(x1, x3)
        y = min(y2, y4) - max(y1, y3)
        return x*y/(abs(x1-x2)*abs(y1-y2)+abs(x3-x4)*abs(y3-y4)-x*y)

# 加载数据
images = []
labels = []
with open('./INRIAPerson/Test/pos.lst', 'r') as f:
    pos_list = f.readlines()
with open('./INRIAPerson/Test/neg.lst', 'r') as f:
    neg_list = f.readlines()
with open('./INRIAPerson/Test/annotations.lst', 'r') as f:
    annotation_list = f.readlines()

for n in range(len(pos_list)):
    pos = pos_list[n]
    pos = pos.strip() #去掉列表中每一个元素的换行符
    pos_img_dir = './INRIAPerson/' + pos
    annotation = annotation_list[n]
    annotation = annotation.strip()
    annotation_dir = './INRIAPerson/' + annotation
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
    # 生成样本框和标签
    img = cv2.imread(pos_img_dir)
    imgs, bounding_boxes = generate_slidingwindow(img)
    for i in range(len(imgs)):
        images.append(imgs[i])
        flag = 0
        for box in groundtruth_boxes:
            if calculate_iou(box, bounding_boxes[i]) > 0.7:
                flag = 1
                break
        labels.append(flag)

for neg in neg_list:
    neg = neg.strip()
    neg_img_dir = './INRIAPerson/' + neg
    img = cv2.imread(neg_img_dir)
    imgs, _ = generate_slidingwindow(img)
    for i in range(len(imgs)):
        images.append(imgs[i])
        labels.append(0)
# 计算Hog特征
hog = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(32, 32), 
                        _blockStride=(16, 16), _cellSize=(16, 16), _nbins=9)
hog_features = []
for img in images:
    hog_feature = hog.compute(img)
    hog_features.append(hog_feature)

# 转换为numpy数组
hog_features = np.array(hog_features, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# 导入logistic回归模型
lr = load('logistic_regression.joblib')
scores = lr.predict_proba(hog_features)[:,1]
scores = np.array(scores, dtype=np.float32)
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC')
plt.legend(loc="lower right")
plt.show()