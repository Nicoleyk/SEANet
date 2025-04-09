# SEANet
## Dependencies and Installation

- Ubuntu 20.04+
- Python 3.8+
- NVIDIA GPU + CUDA 11.8
- PyTorch 2.0+

To install dependencies:
```bash
pip install -r requirements.txt
```
---

## Test

1. [Download the RUOD dataset](https://pan.baidu.com/s/1LXjDZVntddsdlE5-lcS0-w?pwd=5pbd)
2. [Download the pretrained model weights](https://pan.baidu.com/s/1FOB0TxJ0h5EDfhdKGDM9AQ?pwd=nyp6) 


```bash
python val.py --data datasets/ruod.yaml --img 640 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights final/weights/best.pt
```
