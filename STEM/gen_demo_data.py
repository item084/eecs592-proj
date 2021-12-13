import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    re_img = transforms.Resize(imsize)(img)

    return normalize(re_img)

def main():
    image_path = '/home/yuan/projects/mirrorGAN/data/birds/CUB_200_2011/images/010.Red_winged_Blackbird/Red_Winged_Blackbird_0055_4345.jpg'
    imsize = 256
    bbox = [145.0, 88.0, 304.0, 195.0]
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ret = get_imgs(image_path, imsize, bbox, image_transform, norm)
    #ret = get_imgs(image_path, imsize, bbox, None, norm)
    # ret.save("demo.png")
    transforms.ToPILImage()(ret).save("demo.png")
    print(ret)

if __name__ == "__main__":
    main()