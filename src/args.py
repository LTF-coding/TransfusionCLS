import argparse
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser()

    # ===================================== 回归任务 =======================================
    # 数据参数
    parser.add_argument('-l','--location', choices=['all', 'knee', 'hip'] ,help='训练数据属于哪一部位')
    parser.add_argument('--drop_para', type=str, help='选择删除参数,使用逗号隔开')
    parser.add_argument('--drop', choices=['None', 'outliers', '3std'] ,help='删除离群值的方式')
    parser.add_argument('--onehot', action='store_true' ,help='是否使用onehot编码')
    parser.add_argument('--data_name', type=str)

    #
    parser.add_argument('--norm', action='store_true' ,help='是否进行标准化')
    parser.add_argument('--feature_analysis', action='store_true' ,help='是否进行特征分析')

    # 训练参数
    parser.add_argument('--test_ratio', type=float, default=0.15 , help='分割测试集的比例')
    parser.add_argument('--random_state', type=int, default=32 , help='分割测试集的随机种子')
    parser.add_argument('--save_model',  action='store_true' ,help='保存模型')


    # 模型参数
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=40)
    parser.add_argument('--min_samples_leaf', type=int, default=2)
    parser.add_argument('--min_samples_split', type=int, default=2)

    # ===================================== 分类任务 ===========================================
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--load', type=str, help='读取模型的路径')
    parser.add_argument('--focal', dest='focal', action='store_true', help='是否使用focal loss') 
    parser.add_argument('--save_best', dest='save_best', action='store_true', help='是否要保存最优模型') 
    parser.add_argument('--resample', type=int, default=1, help='设置重采样 负例与正例的比例')
    parser.add_argument('--nepoch', type=int, default=40000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--ckpt_dir', type=str, default='/home/ltf/ltf/ML/results/ckpt', help='训练模型的保存路径')
    parser.add_argument('--test_score_dir', type=str, default='/home/ltf/ltf/ML/results/test_score', help='测试结果的保存路径')
    parser.add_argument('--shap', type=str, default='shap_plot_.tiff', help='保存shap图名称,如果有则运行shap')
    parser.add_argument('--model', type=str, default='MLP', help='训练的模型的名称 支持MLP CNN LSTM')



    args = parser.parse_args()
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.test_score_dir):
        os.makedirs(args.test_score_dir)
        
    if args.save_best:
        file_list = os.listdir(args.ckpt_dir)
        pth_files = [ f for f in file_list if f.endswith('.pth')]
        assert pth_files!=None , '保存模型的路径上没有模型(.pth)文件'
    

    return args