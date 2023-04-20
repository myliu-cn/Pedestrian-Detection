# Pedestrian-Detection

## 项目介绍
本项目是人工智能现代方法Ⅱ——机器学习的行人检测作业，具体任务如下所示：
1. 利用HOG特征和SVM方法实现图像中的行人检测
2. 利用逻辑回归 (Logistic Regression)模型实现行人检测，特征不限；并将本方法的结果与任务1中的结果进行比较分析
要求：
1. 利用INRIAPerson Dataset的训练和测试数据进行实验http://pascal.inrialpes.fr/data/human/
2. 画出在Miss Rate(1−Recall) Vs. False Positive Rate 曲线，并计算AUC

## 实验环境  
Python              3.9.16  
matplotlib          3.7.1  
numpy               1.24.2  
opencv-python       4.7.0.72  
scikit-learn        1.2.2  

## 项目内容  
1. 运行前先在代码所在文件夹中创建文件夹INRIAPerson，并将INRIAPerson Dataset中的Train和Test文件夹复制到INRIAPerson文件夹中；
2. 文件夹介绍
SVM_Result           SVM模型运行结果  
Logistic_Result      逻辑回归模型运行结果  
3. 模型文件
svm_model.xml                SVM模型  
logistic_regression.joblib   逻辑回归模型  
4. Code文件内容
>>> **SVM模型**
(1) DataLoader.py  
    内含DataLoader类，类内实现加载所有训练集图片并生成标签；  
    正样本： 把含有人像的部分剪裁下来，并Resize为64*128像素；  
    负样本： 整体Resize为64*128像素并对每张图片随机剪裁（扩充数据集）  
(2) Train.py  
    计算Hog特征并训练SVM模型  
(3) Detector.py  
    实现滑窗、NMS等，对测试样本的行人进行框选  
(4) Test.py  
    加载测试集，用Detector类进行探测  
(5) roc_curve.py  
    绘制ROC曲线，计算AUC  

>>> **Logistic回归模型**  
(1) Logistic_Train.py   
    利用DataLoader加载的数据训练Logistic回归模型  
(2) Logistic_Test.py  
    实现LogisticDetector类并在测试集上测试  
(3) Logistic_roc.py  
    绘制ROC曲线，计算AUC  
