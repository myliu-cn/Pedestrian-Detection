import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from DataLoader import DataLoader
from joblib import dump, load

# 创建逻辑回归对象
lr = LogisticRegression()

# 准备训练数据和标签
data = DataLoader()
images = data.images
labels = data.labels
# 计算Hog特征
hog = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(32, 32), 
                        _blockStride=(16, 16), _cellSize=(16, 16), _nbins=9)
# 计算所有图片的Hog特征
hog_features = []
for img in images:
    hog_feature = hog.compute(img)
    hog_features.append(hog_feature)

# 转换为numpy数组
hog_features = np.array(hog_features, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)
# 训练模型
clf = lr.fit(hog_features, labels)
dump(clf, 'logistic_regression.joblib')
