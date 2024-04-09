import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import os

# test_score文件夹路径
root = '/home/ltf/ltf/ML/results/test_score/test_score_final'
lis = []
for file in os.listdir(root):
    model = file.split('_')[0]
    method = file.split('_')[1]
    # 从文件读取测试结果
    data = np.loadtxt(root+'/'+file, delimiter=',')
    y_true = data[:, 0]
    y_scores = data[:, 1]

    # 计算PR曲线的精确率（Precision）、召回率（Recall）和阈值
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    lis.append((recall,precision,model,pr_auc))


lis.sort(reverse=True, key=lambda x:x[3])
for tup in lis:
    # 绘制PR曲线
    plt.plot(tup[0], tup[1], label=f'{tup[2]}  (AP:{tup[3]:0.2f})')


plt.xlim(0, 1.1)
plt.ylim(0, 1.1)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.grid(True)

# 保存PR曲线图像
plt.savefig('pr_curve.tiff', dpi=600)