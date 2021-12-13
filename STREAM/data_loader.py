import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
# from build_vocab import Vocabulary
# from pycocotools.coco import COCO
import random
import matplotlib.pyplot as plt


class BirdDataset(data.Dataset):
    """CUB Bird Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, fns, caption_vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            caption_vocab: caption and vocabulary
            transform: image transformer.
        """
        self.root = root
        self.fns = fns
        self.caption_vocab = caption_vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        path = self.fns[index] + '.jpg'

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            #img = transforms.ToPILImage()(image)
            #img.show()
            #exit()
            

        random_id = random.randint(0, 9) 

        caption = self.caption_vocab[0][index * 10 + random_id]

        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.fns)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap)for cap in captions] # counting the image embedding
    targets = torch.zeros(len(captions), max(lengths)).long()

    # pad with 0 <end>
    for i, cap in enumerate(captions):
        end = lengths[i]  # counting the image embedding
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, fns, caption_vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # Bird caption dataset
    bird = BirdDataset(root=root,
                       fns=fns,
                       caption_vocab=caption_vocab,
                       transform=transform)
    
    # Data loader for Bird dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=bird, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader