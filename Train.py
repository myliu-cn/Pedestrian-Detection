import cv2
import numpy as np
from DataLoader import DataLoader

data = DataLoader()
# 读取装载好的图片
images = data.images
# 读取装载好的标签
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

# 创建SVM对象并设置参数
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)

# 训练 SVM 模型
svm.trainAuto(hog_features, cv2.ml.ROW_SAMPLE, labels)

# 保存模型
svm.save('svm_model.xml')