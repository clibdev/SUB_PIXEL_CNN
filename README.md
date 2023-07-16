# Fork of [niazwazir/SUB_PIXEL_CNN](https://github.com/niazwazir/SUB_PIXEL_CNN)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.0. (🔥)
* Original pretrained models from GitHub [releases page](https://github.com/clibdev/SUB_PIXEL_CNN/releases). (🔥)
* Installation with [requirements.txt](requirements.txt) file.
* Resaved original model to avoid loading warnings.
* Sample script [test.py](test.py) for inference of single image.

# Installation

```shell
pip install -r requirements.txt
```

# Pretrained models

| Name        | Link                                                                                             |
|-------------|--------------------------------------------------------------------------------------------------|
| SubPixelCNN | [PyTorch](https://github.com/clibdev/SUB_PIXEL_CNN/releases/latest/download/model_epoch_599.pth) |

# Inference

```shell
python test.py --model_path model_epoch_599.pth --input_path data/meerkat.png
```
