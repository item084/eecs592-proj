import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from torchtext.data.metrics import bleu_score
# from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
from data_loader import get_loader 
# from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    # Load files wrapper
    with open(args.testfile_path, 'rb') as f:
        filenames = pickle.load(f)

    # Load vocabulary wrapper
    with open(args.caption_path, 'rb') as f:
        caption_vocab = pickle.load(f)

    models = [[f'models/encoder-{i*5}-70.ckpt', f'models/decoder-{i*5}-70.ckpt'] for i in range(1, 100)]
    models = [
        ['models/old-256-256-1/encoder-235-70.ckpt', 'models/old-256-256-1/decoder-235-70.ckpt'],
        ['models/old-256-512-1/encoder-205-70.ckpt', 'models/old-256-512-1/decoder-205-70.ckpt'],
        ['models/old-256-512-2/encoder-225-70.ckpt', 'models/old-256-512-2/decoder-225-70.ckpt'],
        ['models/old-512-512-1/encoder-310-70.ckpt', 'models/old-512-512-1/decoder-310-70.ckpt'],
        ['models/old-512-512-2/encoder-290-70.ckpt', 'models/old-512-512-2/decoder-290-70.ckpt'],
        ['models/old-512-1024-1/encoder-165-70.ckpt', 'models/old-512-1024-1/decoder-165-70.ckpt'],
        ['models/old-512-1024-2/encoder-140-70.ckpt', 'models/old-512-1024-2/decoder-140-70.ckpt'],
        ['models/old-512-1024-4/encoder-205-70.ckpt', 'models/old-512-1024-4/decoder-205-70.ckpt'],
        ]
    models = [[f'models/old-{args.embed_size}-{args.hidden_size}-{args.num_layers}/encoder-{args.epoch}-70.ckpt', f'models/old-{args.embed_size}-{args.hidden_size}-{args.num_layers}/decoder-{args.epoch}-70.ckpt']]

    for encoder_path, decoder_path in models:
        # Build models
        encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(caption_vocab[2]), args.num_layers)
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # Load the trained model parameters
        # encoder.load_state_dict(torch.load(args.encoder_path))
        # decoder.load_state_dict(torch.load(args.decoder_path))
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))

        def idx2word(word_id):
            return caption_vocab[2][word_id]

        def id2sentence(sent_id):
            sampled_caption = []
            for word_id in sent_id:
                word = idx2word(word_id)
                sampled_caption.append(word)
                if word == '<end>':
                    break
            # sentence = ' '.join(sampled_caption)
            return sampled_caption

        def img2caption(imgpath):
            # Prepare an image
            image = load_image(imgpath, transform)
            image_tensor = image.to(device)
    
            # Generate an caption from the image
            feature = encoder(image_tensor)
            sampled_ids = decoder.sample(feature)
            sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
            # Convert word_ids to words
            return id2sentence(sampled_ids)
    
        cand = []
        ref = []

        for i, filename in enumerate(filenames):
            imgpath = os.path.join(args.imagedir, filename+'.jpg')
            cand.append(img2caption(imgpath))
            ref.append([])
            for j in range(10):
                ref[i].append(id2sentence(caption_vocab[1][i*10+j]))
            # if i % 100 == 0:
            #     print(i)
            # print(cand[i])
            # print(ref[i])
            # print()


        bleu1_score = bleu_score(cand, ref, max_n=1, weights=[0.25] * 1)
        bleu2_score = bleu_score(cand, ref, max_n=2, weights=[0.25] * 2)
        bleu3_score = bleu_score(cand, ref, max_n=3, weights=[0.25] * 3)
        bleu4_score = bleu_score(cand, ref, max_n=4)
        print(f'{args.embed_size}-{args.hidden_size}-{args.num_layers}', bleu1_score, bleu2_score, bleu3_score, bleu4_score)

        # Print out the image and the generated caption
        # print(sentence)
        # image = Image.open(args.image)
        # plt.imshow(np.asarray(image))

def validation(args):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load files wrapper
    with open(args.testfile_path, 'rb') as f:
        filenames = pickle.load(f)

    # Load vocabulary wrapper
    with open(args.caption_path, 'rb') as f:
        caption_vocab = pickle.load(f)
    
    modeldir = "models/"
    eps = [e*20 for e in range(1, 26)]
    eps = [e*10 for e in range(1, 51)]
    eps = [e*5 for e in range(1, 100)]
    
    for e in eps:
        encoder_path = os.path.join(modeldir, f'old-{args.embed_size}-{args.hidden_size}-{args.num_layers}/encoder-{e}-70.ckpt')
        decoder_path = os.path.join(modeldir, f'old-{args.embed_size}-{args.hidden_size}-{args.num_layers}/decoder-{e}-70.ckpt')
        print(encoder_path, decoder_path)

        # Build data loader
        data_loader = get_loader(args.imagedir, filenames, caption_vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=1) 

        # Build the models
        encoder = EncoderCNN(args.embed_size).to(device)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(caption_vocab[2]), args.num_layers).to(device)
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
    
        # Train the models
        total_step = len(data_loader)
        total_loss = 0
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
        
        loss = total_loss / total_step

        # Print log info
        print('Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(loss, np.exp(loss))) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagedir', type=str, default='../data/birds/test_resized', help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-100-70.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-100-70.ckpt', help='path for trained decoder')
    parser.add_argument('--caption_path', type=str, default='../data/birds/bird_captions_mirror.pickle', help='path for vocabulary wrapper')
    parser.add_argument('--testfile_path', type=str, default='../data/birds/test/filenames.pickle', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--batch_size', type=int, default=128) # 128
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors') # 256
    parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states') # 512
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    parser.add_argument('--epoch', type=int , default=1, help='epoch')
    args = parser.parse_args()
    main(args)
    # validation(args)
