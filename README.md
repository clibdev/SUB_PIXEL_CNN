# Fork of [niazwazir/SUB_PIXEL_CNN](https://github.com/niazwazir/SUB_PIXEL_CNN)

Differences between original repository and fork:

* Compatibility with PyTorch >=2.4. (🔥)
* Original pretrained models and converted ONNX models from GitHub [releases page](https://github.com/clibdev/SUB_PIXEL_CNN/releases). (🔥)
* Model conversion to ONNX format using the [export.py](export.py) file. (🔥)
* Installation with [requirements.txt](requirements.txt) file.
* Resaved original model to avoid loading warnings.
* Sample script [test.py](test.py) for inference of single image.

# Installation

```shell
pip install -r requirements.txt
```

# Pretrained models

* Download links:

| Name        | Model Size (MB) | Link                                                                                                                                                                                         | SHA-256                                                                                                                              |
|-------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| SubPixelCNN | 0.2<br>0.2      | [PyTorch](https://github.com/clibdev/SUB_PIXEL_CNN/releases/latest/download/sub-pixel-cnn.pth), [ONNX](https://github.com/clibdev/SUB_PIXEL_CNN/releases/latest/download/sub-pixel-cnn.onnx) | 1b04fcf9c31f69876679b9a91844f2b24ee7346018e130b5e9539fb16bb5ad9c<br>3a66e408a596387192216ea10c6f83b3da73836e7100a3244b2a3aabec001dc7 |

# Inference

```shell
python test.py --model_path sub-pixel-cnn.pth --input_path data/meerkat.jpg
```

# Export to ONNX format

```shell
pip install onnx
```
```shell
python export.py --model_path sub-pixel-cnn.pth --dynamic
```
