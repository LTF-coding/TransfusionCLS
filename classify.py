import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error
import joblib
import shap

import pickle
import xgboost as xgb
import os
import src.args as args

import numpy as np

opt = args.parse_arguments()

train = True

def draw_data_distribution(data, label:str, title:str=''):
    plt.figure(figsize=(12, 8)) 
    plt.hist(data[label])
    plt.xlabel(label)
    plt.ylabel('Frequency')
    if not title:
        plt.title(f'Distribution of {label}')
        plt.savefig(f'./feature_img/Distribution_{label}.png')
    else:
        plt.title(f'Distribution of {title}')
        plt.savefig(f'./feature_img/Distribution_{title}.png')

def wilson_score_interval(accuracy, n_samples, confidence=0.95):
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = accuracy / n_samples
    interval = (p_hat + z*z/(2*n_samples) - z * np.sqrt((p_hat*(1-p_hat)+z*z/(4*n_samples))/n_samples)) / (1 + z*z/n_samples)
    return interval

# 选择特征
features = ['sex', 'age', 'height', 'weight', 'BMI', 'CVD', 'pre-Hb', 'pre-Hct', 'ASA', 'TXA', 'transfusion']

data_knee = pd.read_csv(f'./data/TKA.csv', usecols=features)
data_knee['type'] = 'knee'
data_hip = pd.read_csv(f'./data/THA.csv', usecols=features)
data_hip['type'] = 'hip'
data = pd.concat([data_hip,data_knee])                                   
data['sex'] = data['sex'].replace({'女': 0, '男': 1})

# 检查是否有空值
has_null_values = data.isnull().values.any()

if has_null_values:
    # 输出包含空值的列
    columns_with_null = data.columns[data.isnull().any()].tolist()
    print("Columns with null values:", columns_with_null)

    # 输出包含空值的行
    rows_with_null = data[data.isnull().any(axis=1)]
    print("Rows with null values:")
    print(rows_with_null)
    data = data.dropna()
    print("NaN has droped!")
    # sys.exit()
else:
    print("No null values found in the data.")

print(data.dtypes)



# 划分训练集和测试集
train_data, test_data = train_test_split(data,test_size=0.2, random_state=32)

# 重采样
# 确定多数类和少数类
majority_class = train_data['transfusion'].mode()[0]
minority_class = train_data['transfusion'].unique()[0] if train_data['transfusion'].mode()[0] == train_data['transfusion'].unique()[1] else train_data['transfusion'].unique()[1]
majority_class_data = train_data[train_data['transfusion'] == majority_class]
minority_class_data = train_data[train_data['transfusion'] == minority_class]

# 欠采样：从多数类中随机下采样以匹配少数类的样本数量
downsampled_majority = resample(majority_class_data, 
                                  replace=False,     # 无放回抽样
                                  n_samples=len(minority_class_data),
                                  random_state=32)

# 合并重采样后的多数类数据和少数类数据
train_data = pd.concat([downsampled_majority, minority_class_data])

# 再次检查类别分布，确保重采样后类别平衡
print(data['transfusion'].value_counts())

feature = ['sex', 'age', 'height', 'weight', 'BMI', 'CVD', 'pre-Hb', 'pre-Hct', 'ASA', 'TXA']
target = 'transfusion'

# if 'BMI' in features:
#     train_data['BMI'] = train_data['weight'] / pow(train_data['height'], 2)
#     test_data['BMI'] = test_data['weight'] / pow(test_data['height'], 2)


# # 计算是否需要输血
# train_data['transfusion'] = 1
# train_data.loc[train_data["Post-Hb"] > 100, "transfusion"] = 0
# test_data['transfusion'] = 1
# test_data.loc[test_data["Post-Hb"] > 100, "transfusion"] = 0

print(test_data)

# 统计transfusion
print("train_data transfusion:\n",
      f"0:{len(train_data[train_data['transfusion'] == 0])}",
      f"1:{len(train_data[train_data['transfusion'] == 1])}",
      f"2:{len(train_data[train_data['transfusion'] == 2])}")

print("test_data transfusion:\n",
      f"0:{len(test_data[test_data['transfusion'] == 0])}",
      f"1:{len(test_data[test_data['transfusion'] == 1])}",
      f"2:{len(test_data[test_data['transfusion'] == 2])}")

if train:
    # # ============================= 随机森林 ==================================
    # model = RandomForestClassifier(n_estimators=20,
    #                                 max_depth=10,
    #                                 min_samples_leaf=2,
    #                                 min_samples_split=2)
    # method = 'RF'

    # # ============================= 逻辑回归 ===================================
    # model = LogisticRegression(max_iter=100)
    # method = 'LR'

    # # ============================= 决策树  ====================================
    # model = DecisionTreeClassifier()
    # method = 'DT'

    # ============================= 支持向量机 ==================================
    model = SVC(gamma='scale', decision_function_shape='ovo', probability=False)
    method = 'SVC'

    # # ============================= 朴素贝叶斯 ==================================
    # model = GaussianNB()
    # method = 'NB'

    # # ============================= K最近邻 ==================================
    # model = KNeighborsClassifier()
    # method = 'KNN'

    # 使用训练数据对模型进行训练
    model.fit(train_data[feature], train_data[target])
else:
    # 读取.pkl模型文件
    pkl_path = '/home/ltf/ltf/ML/results/ckpt/ckpt_3-22/SVC_0.9029.pkl'
    model = joblib.load(pkl_path)

# # 绘制SHAP
# # explainer = shap.TreeExplainer(model)
# # shap_values = explainer.shap_values(train_data[feature])
# K = 500  # 设置摘要样本的数量
# data_ = test_data[feature][0:K]
# explainer = shap.KernelExplainer(model.predict, data_)
# shap_values = explainer.shap_values(data_)
# shap.summary_plot(shap_values, data_)

# plt.savefig('shap_plot_KNN.pdf',dpi=800)


# 测试集推理
y_test = test_data[target]
y_pred = model.predict(test_data[feature])
# y_score = model.predict_proba(test_data[feature])[:,1]
# print(y_score)

accuracy = accuracy_score(y_test, y_pred)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
TN, FN, FP, TP = conf_matrix.ravel()
print(TN,FN,FP,TP)
accuracy = (TP + TN)/len(y_test)
ppv = TP / (TP + FP)
npv = TN / (TN + FN)
specificity = TN / (TN + FP)
sensitivity = TP / (TP + FN)

print("Positive Predictive Value (PPV):{:.1f}%".format(ppv*100))
print("Negative Prediction Value (NPV):{:.1f}%".format(npv*100))
print("Specificity:{:.1f}%".format(specificity*100))
print("Sensitivity:{:.1f}%".format(sensitivity*100))
print("Accuracy:{:.3f}".format(accuracy))

# 计算精确度的置信区间
# 法一
z = norm.ppf(1 - (1 - 0.95) / 2)
interval = z* math.sqrt((accuracy*(1-accuracy))/len(y_test))
upper = accuracy + interval
lower = accuracy - interval
# 法二
# print(type(accuracy))
# print(type(y_test))
interval = wilson_score_interval(accuracy, y_test.shape[0])
# print(interval)
lower_bound = accuracy - interval
upper_bound = accuracy + interval

print(f'置信区间：({lower:0.3f}-{upper:0.3f})  ({lower_bound:0.3f}-{upper_bound:0.3f})')

# 保存模型
if train:
    model_name = f"{method}_{accuracy:.4f}.pkl"
    model_path = os.path.join("./results/ckpt/ckpt_SVC", model_name)
    with open(model_path,'wb') as f:
        pickle.dump(model, f)  
    print("\n Model saved successfully.")


# 保存测试结果到文件
if not train:
    method = pkl_path.split('/')[-1].split('_')[0]
# np.savetxt(f'./results/test_score/test_score_resample/{method}_results.txt', np.column_stack((y_test, y_score)), delimiter=',')

