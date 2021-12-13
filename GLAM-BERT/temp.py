from __future__ import print_function

from cfg.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import Trainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

from datasets import prepare_data
from transformers import BertTokenizer, BertModel

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/output/%s_%s_%s' % \
        (cfg.OUTPUT_PATH, cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    data_iter = iter(dataloader)
    data = data_iter.next()
    imgs, captions, cap_lens, class_ids, keys = prepare_data(data)

    text_encoder = BertModel.from_pretrained('bert-base-uncased').cuda()

    sentence_list = []
    for i in range(cfg.TRAIN.BATCH_SIZE):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = self.ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            sentence.append(word)
        sentence=" ".join(sentence)
        sentence_list.append(sentence)
    inputs = tokenizer(sentence, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].cuda()
    sent_emb = text_encoder(**inputs).last_hidden_state.permute(0, 2, 1)
    max_sent_len = max(1, len(max(sent_emb,key=len)))
    words_embs=[]
    for i in sent_emb:
        word_emb = [w for w in i]
        sent_len =  len(i)
        if sent_len < max_sent_len:
            word_emb += [[0]*len(i[0].vector)]*(max_sent_len-sent_len)
        words_embs.append(word_emb)
    # print(len(words_embs),len(words_embs[0]),len(words_embs[0][0]))
    print(word_emb)
    words_embs = torch.Tensor(words_embs).cuda()
                        
        words_embs = words_embs.permute(0,2,1)


from transformers import BertTokenizer, BertModel
text_encoder = BertModel.from_pretrained('bert-base-uncased').cuda()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')