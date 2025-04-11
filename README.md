# SEANet
Underwater object detection is significantly hindered by low-contrast visual conditions and extreme scale variation among marine organisms. To address these challenges, we propose SEANet, a single-stage detection framework specifically tailored for underwater environments. First, the Multi-Scale Detail Amplification Module (MDAM) strengthens feature extraction by expanding receptive fields to capture fine-grained cues in complex backgrounds. Besides, we design the Semantic Enhancement Feature Pyramid (SE-FPN), which incorporates a Fore-Background Contrast Attention (FBC) mechanism. SE-FPN assists in enhancing multi-scale feature integration and moderately improves contrast between targets and their surroundings, helping the network focus more effectively on low-contrast objects in underwater scenes. Experiments on underwater datasets demonstrate that SEANet achieves competitive performance, with the highest AP recorded at 67.0% on the RUOD dataset.

![Detection Results](datasets/show.jpg)

## Install

- Clone this repository
```bash
git clone https://github.com/Nicoleyk/SEANet.git
cd SEANet
```
- Create a conda virtual environment and activate it
```bash
conda create -n seanet python=3.8 +
conda activate seanet
```

- Install required dependencies
```bash
pip install -r requirements.txt
```
- Data Preparation
Download underwater object detection datasets such as [RUOD](https://pan.baidu.com/s/1LXjDZVntddsdlE5-lcS0-w?pwd=5pbd)
It is recommended to extract the datasets outside the project directory. The suggested folder structure is as follows:
```bash
data 
├── images 
│   ├── train 
│   └── val 
├── labels 
│   ├── train 
|   ├── val 
|   ├── train.txt 
└── └── val.txt
```
## Test
  [Download the pretrained model weights](#)  

```bash
python val.py --data datasets/ruod.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights final/weights/best.pt
```
