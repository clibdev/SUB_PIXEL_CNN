import numpy as np
import torch
from torch.nn import init
from torchvision import transforms
from torch import nn
from PIL import Image
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='./data/meerkat.jpg')
parser.add_argument('--output_dir', default='./runs')
parser.add_argument('--model_path', default='./model_epoch_599.pth')
parser.add_argument('--device', default='cuda', help='cuda or cpu')
opt = parser.parse_args()


class Network(nn.Module):
    def __init__(self, upscale_factor):
        super(Network, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


device = torch.device(opt.device)

img = Image.open(opt.input_path)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

img_y = transforms.ToTensor()(img_y).unsqueeze_(0)

model = torch.load(opt.model_path, weights_only=False).to(device)
model.eval()

with torch.no_grad():
    output = model(img_y.to(device))
    image_out = output[0].cpu().numpy()

    out_img_y = image_out * 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    final_img = Image.merge('YCbCr', [out_img_y,
                                      img_cb.resize(out_img_y.size, Image.BICUBIC),
                                      img_cr.resize(out_img_y.size, Image.BICUBIC),
                                      ]).convert('RGB')

    final_img.save(os.path.join(opt.output_dir, os.path.basename(opt.input_path)))
