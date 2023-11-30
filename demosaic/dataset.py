import glob
import random

from PIL import Image, ImageDraw, ImageFilter
from torch.utils import data
from torchvision import transforms

path = "G:\datasets\lbxx\lbxx"


class ReConstructionDataset(data.Dataset):
    def __init__(self, data_dir=r"G:\datasets\lbxx\lbxx", image_size=64):
        self.image_size = image_size
        # 预处理
        self.trans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.image_paths = glob.glob(r"G:\datasets\lbxx\lbxx\*")

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        return self.trans(self.create_blur(image)), self.trans(image)

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def create_blur(image, box_size=200):
        mask = Image.new('L', image.size, 255)
        draw = ImageDraw.Draw(mask)
        upper_left_corner = (
            random.randint(0, image.size[0] - box_size),
            random.randint(0, image.size[1] - box_size)
        )
        lower_right_corner = (
            upper_left_corner[0] + box_size,
            upper_left_corner[1] + box_size
        )
        draw.rectangle([*upper_left_corner, *lower_right_corner], fill=0)
        masked_image = Image.composite(image, image.filter(ImageFilter.GaussianBlur(15)), mask)
        return masked_image


if __name__ == '__main__':
    image = Image.open(r"G:\datasets\lbxx\lbxx\10499.jpeg")
    ReConstructionDataset.create_blur(image)
