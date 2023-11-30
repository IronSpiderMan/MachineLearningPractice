import cv2
import torch
from PIL import Image
from torch.utils import data
from torchvision.utils import make_grid

from demosaic.model import ReConstructionNetwork
from demosaic.dataset import ReConstructionDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = data.DataLoader(ReConstructionDataset(), 64)
unet = ReConstructionNetwork().to(device)
unet.load_state_dict(torch.load('reconstruction.pth'))
for masked_images, images, in dataloader:
    masked_images, images = masked_images.to(device), images.to(device)
    with torch.no_grad():
        outputs = unet(masked_images)
        outputs = torch.concatenate((images, masked_images, outputs), dim=-1)
        outputs = make_grid(outputs)
        img = outputs.cpu().numpy().transpose(1, 2, 0)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        Image.fromarray(img).show()
    break
