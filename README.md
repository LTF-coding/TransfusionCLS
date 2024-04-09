# TransfusionCLS
*Preoperative Patient Characteristics Predicts Blood Transfusion Requirements Following Total Knee or Hip Arthroplasty Independent of Intraoperative and Postoperative Characteristics: A National Case-control Study Based on Deep Learning Algorithms*

## Installation
1. Clone the repository locally
```bash
git clone https://github.com/LTF-coding/TransfusionCLS.git
cd TransfusionCLS
```
2. It is recommended that you create a virtual environment using python 3.10 or newer
```bash
conda create -n TransCls python=3.10
```
3. Use the following command to install the dependencies
```bash 
pip install -r requirements.txt
```

## Usage
### ML

For machine learning algorithms, it is necessary to change the source code to complete the training and testing

```bash 
python classify.py
```

### DL

The following parameters can be used for testing and training deep learning algorithms

1. Train
```bash 
python classify_DL.py  --train --nepoch 40000 --lr 0.0001 --resample 0 --model MLP --ckpt_dir './results/ckpt' --test_score_dir './results/test_score' --shap 'shap_plot.tiff' 
```
+ `train`: Execute this parameter representation for training otherwise for testing
+ `nepoch`: Num of epoch
+ `lr`: Learning rate
+ `resample`: Rate of resampling
+ `model`:Choose a model to train from MLP,CNN,LSTM
+ `shap`: Whether to save the shap graph and where to save it
+ `ckpt_dir`： The path of where the model is saved
+ `test_score_dir`: The path of where the test score is saved

Or you can save the command in `cls.sh`
```bash
bash cls.sh
```

2. Test

It will be tested by default. If you want to test again with an existing model, you can use the following command
```bash
python classify_DL.py  --resample 0 --model MLP --test_score_dir './results/test_score' --shap 'shap_plot.tiff' --load 'path to .pth' 
```
`reload`: The path of the model to be tested
