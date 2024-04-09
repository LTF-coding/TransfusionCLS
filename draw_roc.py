import numpy as np
from sklearn.metrics import roc_curve, auc
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

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    lis.append((fpr,tpr,model,roc_auc))
    # lis.append((fpr,tpr,model,method,roc_auc))


lis.sort(reverse=True, key=lambda x:x[3])
for tup in lis:
    # 绘制PR曲线
    plt.plot(tup[0], tup[1], label=f'{tup[2]} (AUC:{tup[3]:0.2f})')


plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)

# 保存ROC曲线图像
plt.savefig('roc_curve.tiff', dpi=600)
