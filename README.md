# 介绍

yolov5进行佩戴口罩识别，并利用fastapi进行web端的部署

# 训练

> 训练可以在`个人电脑（如果性能好的话）` ，其它：`GoogleColab` 、 [Openbayes](https://openbayes.com) 或者其它更多的GPU训练平台。

1. 下载[yolov5](https://github.com/ultralytics/yolov5)

2. 下载Mask数据集：[下载1](https://public.roboflow.ai/object-detection/mask-wearing) 、 [下载2](https://download.csdn.net/download/djstavaV/12624588) 、 [下载3（wja4）](https://pan.baidu.com/s/15GSPiJ59dg4kNyUch6W5Xw)

3. 修改`yolov5/models/yolov5s.yaml`，将原来的`nc: 80`改为`nc: 2`

> `yolov5`有`yolov5s.yaml` 、 `yolov5m` 、`yolov5l` 、`yolov5x`等几个模型，这里我们采用最轻量级的`5s`，所以修改`5s`的配置文件。关于为什么改成2？因为我们的口罩数据集只有`mask`和`no-mask`两个分类。

4. 训练：
```python
!python train.py --data ../mask/data.yaml --cfg models/yolov5s.yaml --weights '' --batch-size 64
```

> 在`GoogleColab`上运行会报一个错，我们升级下相应模块就可以：
```python
pip install -U pyyaml
```

5. 模型测试：
```python
$ python detect.py --weights 你的模型文件保存位置 --source 0  # webcam
						file.jpg  # image 
						file.mp4  # video
						path/  # directory
						path/*.jpg  # glob
						rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
						rtmp://192.168.1.105/live/test  # rtmp stream
						http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
```

# 部署

安装必要的包：
```python
fastapi
aiofiles
uvicorn
python-multipart
```

> 部署的代码包括`yolov5/server.py` 、 `yolov5/predict.py` 、 `yolov5/inference` ，部署代码一点要放在`yolov5`文件夹下。修改部署代码的模型位置既可以。

```python
python server.py
```

# 参考

- 使用google colab训练YOLOv5模型：https://xugaoxiang.com/2020/11/01/google-colab-yolov5/
- yolov5：https://github.com/ultralytics/yolov5
- fastapi部署yolov5：https://github.com/Frank1126lin/deploy_yolov5_with_fastapi



