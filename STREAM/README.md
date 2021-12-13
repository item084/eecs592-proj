# Image Captioning

## Usage

### 1. Resize the images

```bash
python resize.py
```

### 2. Train the model

```bash
python train.py    
```

### 3. Test the model

```bash
python3 test.py --embed_size=256 --hidden_size=256 --num_layers=1 --epoch=235;
python3 test.py --embed_size=256 --hidden_size=512 --num_layers=1 --epoch=205;
python3 test.py --embed_size=256 --hidden_size=512 --num_layers=2 --epoch=225;
python3 test.py --embed_size=512 --hidden_size=512 --num_layers=1 --epoch=310;
python3 test.py --embed_size=512 --hidden_size=512 --num_layers=2 --epoch=290;
python3 test.py --embed_size=512 --hidden_size=1024 --num_layers=1 --epoch=165;
python3 test.py --embed_size=512 --hidden_size=1024 --num_layers=2 --epoch=140;
python3 test.py --embed_size=512 --hidden_size=1024 --num_layers=4 --epoch=205
python sample.py --image='png/example.png'
python sample.py --image='/home/yuan/projects/mirrorGAN/data/birds/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
python sample.py --image='/home/yuan/projects/mirrorGAN/data/birds/CUB_200_2011/images/165.Chestnut_sided_Warbler/Chestnut_Sided_Warbler_0103_163669.jpg'
python sample.py --image='/home/yuan/projects/mirrorGAN/data/birds/CUB_200_2011/images/098.Scott_Oriole/Scott_Oriole_0002_795829.jpg'
python sample.py --image='/home/yuan/projects/mirrorGAN/data/birds/CUB_200_2011/images/035.Purple_Finch/Purple_Finch_0025_28174.jpg'

```

