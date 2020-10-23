### 1. Training Classifier-Baseline
python train_classifier.py --config configs/train_classifier_mini.yaml


### 2. Training Meta-Baseline
python train_meta.py --config configs/train_meta_mini.yaml

### 3. Test
To test the performance, modify `configs/test_few_shot.yaml` by setting `load_encoder` to the saving file of Classifier-Baseline, or setting `load` to the saving file of Meta-Baseline.

E.g., `load: ./save/meta_mini-imagenet-1shot_meta-baseline-resnet12/max-va.pth`

Then run:
python test_few_shot.py --shot 1

### 4. Visualization
tensorboard --logdir=./save/classifier_mini-imagenet_resnet12/tensorboard
tensorboard --logdir=./save/meta_mini-imagenet-1shot_meta-baseline-resnet12/tensorboard


### 5.Experiments
train_classifier.py  #  Classifier-Baseline
train_meta.py        #  Meta-Baseline

train_ssl_co.py      # add MoCo between Classifier and Meta Baseline
train_ssl_byol.py    # add BYOL between Classifier and Meta Baseline
train_meta_ssl       # Meta-Learning before SSL

train_meta_co.py      # joint MoCo in Meta Baseline
train_meta_byol.py    # joint BYOL in Meta Baseline