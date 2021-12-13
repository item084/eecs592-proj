import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
# from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load files wrapper
    with open(args.trainfile_path, 'rb') as f:
        filenames = pickle.load(f)

    # Load vocabulary wrapper
    with open(args.caption_path, 'rb') as f:
        caption_vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, filenames, caption_vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(caption_vocab[2]), args.num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (epoch+1) % args.save_epoch == 0 and (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, f'old-{args.embed_size}-{args.hidden_size}-{args.num_layers}/decoder-{epoch+1}-{i+1}.ckpt'))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, f'old-{args.embed_size}-{args.hidden_size}-{args.num_layers}/encoder-{epoch+1}-{i+1}.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../data/birds/train_resized', help='directory for resized images')
    parser.add_argument('--trainfile_path', type=str, default='../data/birds/train/filenames.pickle', help='path for train annotation json file')
    parser.add_argument('--caption_path', type=str, default='../data/birds/bird_captions_mirror.pickle', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_epoch', type=int , default=5, help='epoch size for saving trained models')
    parser.add_argument('--save_step', type=int , default=70, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors') # 256
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states') # 512
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm') # 1
    
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)#default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)

# added length of feature / end
# added drop in embedding
# added drop in lstm