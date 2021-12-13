import argparse
import os
from PIL import Image
import pickle


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, num_images, output_dir))

def resize_images_bird_train(image_dir="../data/birds/CUB_200_2011/images", output_dir="../data/birds/train_resized", size=[256, 256]):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open('../data/birds/train/filenames.pickle', 'rb') as f:
        fns = pickle.load(f, encoding='latin1')
    for i, image in enumerate(fns):
        image = image + ".jpg"
        imgprefix = image.split('/')[0]
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                if not os.path.exists(os.path.join(output_dir, imgprefix)):
                    os.makedirs(os.path.join(output_dir, imgprefix))
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, len(fns), output_dir))

def resize_images_bird_test(image_dir="../data/birds/CUB_200_2011/images", output_dir="../data/birds/test_resized", size=[256, 256]):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open('../data/birds/test/filenames.pickle', 'rb') as f:
        fns = pickle.load(f, encoding='latin1')
    for i, image in enumerate(fns):
        image = image + ".jpg"
        imgprefix = image.split('/')[0]
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                if not os.path.exists(os.path.join(output_dir, imgprefix)):
                    os.makedirs(os.path.join(output_dir, imgprefix))
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print ("[{}/{}] Resized the images and saved into '{}'."
                   .format(i+1, len(fns), output_dir))

def main(args):
    image_dir = args.image_dir
    output_dir = args.output_dir
    image_size = [args.image_size, args.image_size]
    resize_images_bird_test()
    # resize_images_bird_train()
    # resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/train2014/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./data/resized2014/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)