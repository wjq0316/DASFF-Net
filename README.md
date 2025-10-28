# DASFF-Net: Spatial-Frequency Fusion with Depth Awareness for Camouflaged Object Detection

# 1. Requirements
## 1.1 Configuring your environment (Prerequisites):
- Creating a virtual environment in terminal: `conda create -n DASFFNet python=3.6.13`.
- Installing necessary packages: `pip install -r requirements.txt`.

## 1.2 Downloading necessary data:
- downloading testing dataset and move it into ./CodDataset/TestDataset/, which can be found in this download link (https://pan.baidu.com/s/12lYTC_YPb4jVZdM4l1rK9Q  Code: 1234).
- downloading training dataset and move it into ./CodDataset/TrainDataset/, which can be found in this download link (https://pan.baidu.com/s/12lYTC_YPb4jVZdM4l1rK9Q  Code: 1234).
- downloading smt pretrained weights and move it into ./smt_tiny.pth, which can be found in this download link (https://pan.baidu.com/s/12lYTC_YPb4jVZdM4l1rK9Q  Code: 1234).

## 1.3 Training Configuration:
- Set parameters and the save path (./ckps/DASFFNet/) in the options_cod.py file, then run the PEMFNet_train_cod.py file.

## 1.4 Testing Configuration:
- After you download all the pre-trained model and testing dataset, just run PEMFNet_test_cod.py to generate the final prediction map.

## 1.5 Evaluating your trained model:
Assigning your costumed path, like method, mask_root and pred_root in eval.py. Just run eval.py to evaluate the trained model.

# 2. Our Results
- The pre-computed maps of DASFF-Net can be found in this download link (https://pan.baidu.com/s/13NXKz1ID2KexKJ630HyOuQ 提取码: 1234).
- The best pretrained weights of DASFF-Net can be found in this download link (https://pan.baidu.com/s/13NXKz1ID2KexKJ630HyOuQ 提取码: 1234).


# Contact
Feel free to send e-mails to me (2669061453@qq.com).<br>


