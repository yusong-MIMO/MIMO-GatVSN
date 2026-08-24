# MIMO-GatVSN

Official implementation of MIMO-GatVSN for video steganography.

---

## Dependencies and Installation

```bash
pip install -r requirements.txt
```

## Data Preparing

 Please download the training and evaluation dataset from [Vimeo-90K](http://toflow.csail.mit.edu/). 

## Train

```
cd code
python train.py -opt options/train/train_MIMO_1video.yml
```

## Test

```
cd code
python test.py -opt options/train/train_MIMO_1video.yml
```

