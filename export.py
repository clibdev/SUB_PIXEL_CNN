import argparse
import os
import torch
from torch import nn
from torch.nn import init


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='./model_epoch_599.pth', help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='cuda or cpu')
    parser.add_argument('--dynamic', action='store_true', default=False, help='enable dynamic axis in onnx model')
    opt = parser.parse_args()

    model = torch.load(opt.model_path).to(opt.device)
    model = model.to(opt.device)
    model.eval()

    model_path = os.path.splitext(opt.model_path)[0] + '.onnx'

    dummy_input = torch.randn(1, 1, 224, 224).to(opt.device)
    dynamic_axes = {'input': {2: '?', 3: '?'}, 'output': {2: '?', 3: '?'}} if opt.dynamic else None
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=17
    )
