import torch
from torch import nn
from torch import optim
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from demosaic.model import ReConstructionNetwork
from demosaic.dataset import ReConstructionDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, dataloader, optimizer, criterion, epochs):
    model = model.to(device)
    for epoch in range(epochs):
        for iter, (masked_images, images) in enumerate(dataloader):
            masked_images, images = masked_images.to(device), images.to(device)
            outputs = model(masked_images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (iter + 1) % 100 == 1:
                print("epoch: %s, iter: %s, loss: %s" % (epoch + 1, iter + 1, loss.item()))
                with torch.no_grad():
                    outputs = make_grid(outputs)
                    img = outputs.cpu().numpy().transpose(1, 2, 0)
                    plt.imshow(img)
                    plt.show()
    torch.save(model.state_dict(), './reconstruction.pth')


if __name__ == '__main__':
    dataloader = data.DataLoader(ReConstructionDataset(), 64)
    unet = ReConstructionNetwork()
    unet.load_state_dict(torch.load('reconstruction.pth'))
    optimizer = optim.Adam(unet.parameters(), lr=0.0002)
    criterion = nn.MSELoss()
    train(unet, dataloader, optimizer, criterion, 20)
