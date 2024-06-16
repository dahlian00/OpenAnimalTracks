
## Requirement
```python
pip install -r requirements.txt
```

## Training
1. First, please set a variable ```basedir``` in ```train.py```.
2. Train models by following commands.

- VGG16
```python
python3 train.py config/vgg16.yaml --use_adj
```

- ResNet-50
```python
python3 train.py config/resnet50.yaml --use_adj
```

- EfficientNet-B1
```python
python3 train.py config/efficientnet_b1.yaml --use_adj
```

- ViT-B
```python
python3 train.py config/vit_b.yaml --use_adj
```

- SwinTransformer-B
```python
python3 train.py config/swin_b.yaml --use_adj
```




## Testing
- Faster RCNN
- VGG16
```python
python3 test.py config/vgg16.yaml -w /path/to/weight.pth -n 1
```

- ResNet-50
```python
python3 test.py config/resnet50.yaml -w /path/to/weight.pth -n 1
```

- EfficientNet-B1
```python
python3 test.py config/efficientnet_b1.yaml -w /path/to/weight.pth -n 1
```

- ViT-B
```python
python3 test.py config/vit_b.yaml -w /path/to/weight.pth -n 1
```

- SwinTransformer-B
```python
python3 test.py config/swin_b.yaml -w /path/to/weight.pth -n 1
```