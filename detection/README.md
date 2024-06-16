
## Requirement
Please install ```MMDetection v3.1.0 ``` manually.  
Then run the following command:
```python
pip install -r requirements.txt
```

## Preparation
Please the files of ```config_files``` into the corresponding directory of ```mmdetection/configs/```.

## Training
1. Set a variable ```basedir``` in all the files of ```config_files```.
2. Move to the ```mmdetection``` detectory.
3. Train models by following commands.

- FasterRCNN
```python
python3 tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_ms-3x_oat.py
```

- SSD
```python
python3 tools/train.py configs/ssd/ssdoat.py 
```

- YOLOX
```python
python3 tools/train.py configs/yolox/yolox_s_oat.py 
```



## Testing
- Faster RCNN
```python
python3 tools/test.py configs/faster_rcnn/faster-rcnn_r50_fpn_ms-3x_oat.py work_dirs/faster-rcnn_r50_fpn_ms-3x_oat/epoch_24.pth --show-dir results/fasterrcnn
```


- SSD
```python
python3 tools/test.py configs/ssd/ssdoat.py work_dirs/ssdoat/epoch_24.pth --show-dir results/ssd
```

- YOLOX
```python
python3 tools/test.py configs/yolox/yolox_s_oat.py  work_dirs/yolox_s_oat/epoch_100.pth --show-dir results/yolox
```