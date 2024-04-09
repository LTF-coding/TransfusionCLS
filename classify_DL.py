import sys
import copy
import pandas as pd
import matplotlib.pyplot as plt
from src.focal_loss import FocalLoss
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
import torch.optim.lr_scheduler as lr_scheduler
import seaborn as sns
import torch
import torch.nn as nn
from src.linear import LinearNet
from src.conv1d import ConvModel
from src.LSTM import LSTM
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import math
from scipy.stats import norm
import shap
import os 
import src.args as args

opt = args.parse_arguments()

torch.manual_seed(42)

# 选择特征
features = ['sex', 'age', 'height', 'BMI', 'weight', 'CVD', 'pre-Hb', 'pre-Hct', 'ASA', 'TXA', 'transfusion']
data_knee = pd.read_csv(f'./data/TKA.csv', usecols=features)
data_knee['type'] = 'knee'
data_hip = pd.read_csv(f'./data/THA.csv', usecols=features)
data_hip['type'] = 'hip'
features.append('type')
data = pd.concat([data_hip,data_knee])
data['sex'] = data['sex'].replace({'女': 0, '男': 1})

# 判断是否有空行
has_null_values = data.isnull().values.any()
if has_null_values:
    columns_with_null = data.columns[data.isnull().any()].tolist()
    print("Columns with null values:", columns_with_null)
    rows_with_null = data[data.isnull().any(axis=1)]
    print("Rows with null values:")
    print(rows_with_null)
    data = data.dropna()
    print("NaN values have been dropped!")
else:
    print("No null values found in the data.")



# 对分类变量进行OneHot编码
cat_cols = ['sex', 'type', 'transfusion']
onehot_cols = [['male','female'],['hip','knee'], ['no','yes']]
le = LabelEncoder()
for col in cat_cols:
    data.loc[:,col] = le.fit_transform(data[col])

for i,col in enumerate(cat_cols):
    cat_concat = data[col].to_numpy().reshape((-1,1))
    onehot = OneHotEncoder()
    classes = onehot.fit_transform(cat_concat).toarray()
    for j,col in enumerate(onehot_cols[i]):
        data[col] = classes[:,j]

data = data.astype({'age':int,
                    'male':int, 'female':int,
                    'hip':int, 'knee':int,
                    'no':int, 'yes':int})

# 更新选择参数
onehot_cols.remove(['no','yes'])
# cat_cols.remove('transfusion')
for col in cat_cols:
    features.remove(col)
for col in onehot_cols:
    for subcol in col:
        features.append(subcol)
# features.append('tourniquet')
print(features)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=32)
# 重采样
if opt.resample != 0:
    majority_class_data = train_data[train_data['yes'] == 0]
    minority_class_data = train_data[train_data['yes'] == 1]
    # 欠采样：从多数类中随机下采样以匹配少数类的样本数量
    downsampled_majority = resample(majority_class_data, 
                                    replace=False,     # 无放回抽样
                                    n_samples=len(minority_class_data) * opt.resample,  # 设置负例与正例的比例
                                    random_state=2397)

    # 合并重采样后的多数类数据和少数类数据
    train_data = pd.concat([downsampled_majority, minority_class_data])

    # 再次检查类别分布，确保重采样后类别平衡
    print(train_data['yes'].value_counts())


print('train_data:',train_data)
print('test_data:',test_data)

# Convert the data to PyTorch tensors
target = ['no','yes']
X_train = torch.tensor(train_data[features].values, dtype=torch.float32)
y_train = torch.tensor(train_data[target].values, dtype=torch.float32)
X_test = torch.tensor(test_data[features].values, dtype=torch.float32)
y_test = torch.tensor(test_data[target].values, dtype=torch.float32)
print('X_train:',X_train.shape)

if opt.model=='MLP':
    model = LinearNet(X_train.shape[1], len(set(data['transfusion'])))
elif opt.model=='CNN':
    model = ConvModel(1,2,32)
elif opt.model=='LSTM':
    model = LSTM(X_train.shape[1],hidden_size=64,num_layers=5,num_classes=2,data_len=X_train.shape[0])
else:
    print('error model')
    sys.exit()
criterion = FocalLoss(10,2) if opt.focal else nn.CrossEntropyLoss()  


# 训练
if opt.train:
    # Define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # 定义学习率衰减策略
    scheduler = lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.5)
    train_label = torch.tensor(train_data['transfusion'].values)
    train_target = torch.cat((torch.tensor(train_data['no'].values).unsqueeze(1),
                              torch.tensor(train_data['yes'].values).unsqueeze(1)),dim=1).to(torch.float32)
    target = train_target if opt.focal else train_label 
    if type(model).__name__=='LSTM':
        X_train = X_train.unsqueeze(1)
        X_test = X_test.unsqueeze(1)
    
    # Train the neural network
    num_epochs = opt.nepoch
    losses = []

    # 获取当前保存的最高精度 
    train_acc_list = []
    for file in os.listdir(opt.ckpt_dir):
        acc = file.split('_')[-1].split('.')[1]
        acc = '0.' + acc
        train_acc_list.append(acc)
    train_acc_list = [float(i) for i in train_acc_list]
    train_best = np.array(max(train_acc_list)) if len(train_acc_list)!=0 else 0
    print(train_best)


    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, target.long())
        losses.append(loss.detach().numpy())
        y_hat = copy.copy(outputs)
        TP = torch.sum(train_label.flatten() == torch.argmax(y_hat, dim=1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        acc = TP.data.numpy()/len(X_train)
        print('epoch:',epoch,f'lr:{lr:.5}','loss:',loss.item(),'acc:',round(acc,3),f'TP:{TP}/{len(X_train)}')
        if acc > train_best and opt.save_best:
            train_best = acc
            torch.save(model.state_dict(), f'{opt.ckpt_dir}/MLP_{acc}.pth')

    train_best = acc if not opt.save_best else train_best
    torch.save(model.state_dict(), f'{opt.ckpt_dir}/MLP_final_{acc}.pth')
    
else:
    print('load')
    model.load_state_dict(torch.load(opt.load))


# Test the neural network
test_label = torch.tensor(test_data['transfusion'].values)
test_target = torch.cat((torch.tensor(test_data['no'].values).unsqueeze(1),
                         torch.tensor(test_data['yes'].values).unsqueeze(1)),dim=1).to(torch.float32)
target = test_target if opt.focal else test_label 

model.eval()
if type(model).__name__=='LSTM':
    X_test = X_test.unsqueeze(1)
    X_train = X_train.unsqueeze(1)
    train_label = torch.tensor(train_data['transfusion'].values)
    # l = X_test.shape[0]
    # X_test = torch.cat((X_test,X_train[0:21222-l]))
    # target = torch.cat((test_label,train_label[0:21222-l]))
    l = X_train.shape[0]
    X_test = X_test[0:l]
    target = test_label[0:l]
    test_label = test_label[0:l]
    print(X_test.shape)

with torch.no_grad():
    y_pred = model(X_test)
    loss = criterion(y_pred, target)
    # 将预测结果转换为类别标签
    y_pred_labels = torch.argmax(y_pred, dim=1)
    if type(model).__name__=='LSTM':
        y_pred_labels = y_pred_labels[0:l]
    TP = torch.sum((y_pred_labels == 1) & (test_label == 1))
    TN = torch.sum((y_pred_labels == 0) & (test_label == 0))
    FP = torch.sum((y_pred_labels == 1) & (test_label == 0))
    FN = torch.sum((y_pred_labels == 0) & (test_label == 1))
    accuracy = (TP.data.numpy() + TN.data.numpy())/len(X_test)
    if type(model).__name__=='LSTM':
        accuracy = (TP.data.numpy() + TN.data.numpy())/l

if opt.shap!='':    
    # 绘制 SHAP 图
    print('start SHAP ...')
    print(X_test.shape)
    K = 1000 # 设置摘要样本的数量
    X_test = X_test[0:K]
    explainer = shap.DeepExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test)
    a = np.array(shap_values)
    _,_,n = a.shape
    shap_values = np.zeros((2,K,n-2))
    shap_values[:,:,0:n-4] = a[:,:,0:n-4]
    shap_values[:,:,n-4] = (a[:,:,n-4] + a[:,:,n-3]) / 2
    shap_values[:,:,n-3] = (a[:,:,n-2] + a[:,:,n-1]) / 2
    shap_feature = ['Age', 'Height', 'Weight', 'BMI', 'CVD', 'Pre-Hb', 'Pre-Hct', 'ASA', 'TXA', 'Sex', 'Type of Surgery']
    X_test_ = torch.cat((X_test[:,0:n-3],X_test[:,n-2].unsqueeze(1)),dim=1)
    shap.summary_plot(shap_values[1], X_test_, feature_names=shap_feature)

    plt.savefig(opt.shap, dpi=600)

# 计算四个指标
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
z = norm.ppf(1 - (1 - 0.95) / 2)
interval = z* math.sqrt((accuracy*(1-accuracy))/len(y_test))
upper = accuracy + interval
lower = accuracy - interval

print(f'置信区间：({lower:0.3f}-{upper:0.3f})')

if type(model).__name__=='LSTM':
    y_pred = y_pred[0:l]

# 使用softmax函数将得分转换为概率
probabilities = F.softmax(y_pred, dim=1)
print(y_pred.shape)


# 保存测试结果到文件
np.savetxt(f'{opt.test_score_dir}/MLP_results.txt', np.column_stack((test_label, probabilities[:,1])), delimiter=',')

