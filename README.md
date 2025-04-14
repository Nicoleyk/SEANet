# SEANet
Underwater object detection is significantly hindered by low-contrast visual conditions and extreme scale variation among marine organisms. To address these challenges, we propose SEANet, a single-stage detection framework specifically tailored for underwater environments. First, the Multi-Scale Detail Amplification Module (MDAM) strengthens feature extraction by expanding receptive fields to capture fine-grained cues in complex backgrounds. Besides, we design the Semantic Enhancement Feature Pyramid (SE-FPN), which incorporates a Fore-Background Contrast Attention (FBC) mechanism. SE-FPN assists in enhancing multi-scale feature integration and moderately improves contrast between targets and their surroundings, helping the network focus more effectively on low-contrast objects in underwater scenes. Experiments on underwater datasets demonstrate that SEANet achieves competitive performance.

<p align="center">
  <img src="datasets/show.jpg" width="600"/>
</p>

🧱 Project Structure and Modules
🔧 File Structure

```bash
SEANet/
├── models/             # All core model modules
│   ├── detect/         # seanet.yaml
│   ├── common.py       # Backbone components and blocks
│   ├── GFPN/ # Custom modules like FBC, MDAM
├── train.py            # Training pipeline
├── val.py              # Evaluation script
├── datasets/           # Dataset yaml files
└── utils/              # Helper functions, logging, plotting
```

📌 Key Modules (paper vs. code naming)
Paper Name	Code Implementation (File)
MDAM (Multi-Scale Detail Amplification Module)	RF_D_gn in models/GFPN/RF_gn.py
CEM (Contrast Enhancement Module)	CSPStageEM in models/GFPN/Enhancement_module.py
FBC (Fore-Background Contrast)	FBSIE in models/GFPN/Enhancement_module.py
SE-FPN	Implemented in models/detect/seanet.yaml as feature fusion hierarchy


## 🚀 Installation

- Clone this repository
```bash
git clone https://github.com/Nicoleyk/SEANet.git
cd SEANet
```
- Create a conda virtual environment and activate it
```bash
conda create -n seanet python=3.8+
conda activate seanet
```
- Install required dependencies
```bash
pip install -r requirements.txt
```
## 📂 Data Preparation
Download underwater object detection datasets such as [RUOD](https://pan.baidu.com/s/165NIEGmyHIVeCy47WIF8LA?pwd=w35g)
The suggested folder structure is as follows:
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
## 🧪 Evaluation
- [Download the pretrained model weights](https://pan.baidu.com/s/1pDGsseIr2M4b0sYFWN8ALg?pwd=9abj) and place it in:
```bash
runs/train/final/
```
- Run evaluation:
```bash
python val.py --data datasets/ruod.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights runs/train/final/weights_ruod/best.pt
```
## 🔧 Robustness Evaluation
We also provide robustness subsets of RUOD(Gaussian Noise and Motion Blur at 5 severity levels),you can download them from [RUOD]( https://pan.baidu.com/s/165NIEGmyHIVeCy47WIF8LA?pwd=w35g )

![Detection Results](datasets/robustness_show_00.png)

```bash
# Evaluate on Gaussian Noise
python val.py --data datasets/ruod-gussian.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights runs/train/final/weights_ruod/best.pt

# Evaluate on Motion Blur
python val.py --data datasets/ruod-motionblur.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights runs/train/final/weights_ruod/best.pt

```

## 🧪 Train
```bash
nohup python -u train.py --workers 4 --batch 16 --data datasets/yourdataset.yaml --img 640 \
--cfg models/detect/seanet.yaml --weights '' --hyp hyp.scratch-high.yaml --epochs 300 --close-mosaic 10 &

```

## 📚 Citation
If you find this project useful in your research, please cite:
```bash
@misc{yang2025seanet,
  title     = {Advanced Semantic Amplification for Underwater Object Detection in Low-Contrast and Multi-Scale},
  note      = {The Visual Computer (under review)},
  year      = {2025},
  howpublished = {\url{https://github.com/Nicoleyk/SEANet}}
}
```
