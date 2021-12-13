# GLAM-BERT

## steps

### Requirements

- torch-1.10.0
- pandas-1.3.4
- torchvision-0.11.1
- nltk-3.6.5
- easydict-1.9
- pyyaml-6.0
- scikit-image-0.18.3

### Run

``` bash
cfg=cfg/train_bird.yml
python3 main.py --cfg 'cfg/train_bird_bert.yml'
python3 main.py --cfg 'cfg/eval_bird_bert.yml'
python3 main.py --cfg 'cfg/sample_bird_bert.yml'
```
