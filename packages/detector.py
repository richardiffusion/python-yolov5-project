import torch
import numpy as np
from PIL import Image
from .models import Detection

class YOLOv5Detector:
    def __init__(self, weights_path='weights/yolov5s.pt', conf_thres=0.4, iou_thres=0.6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_labels = ["figure"] # 对应 Swift
        self.colors = [(0, 122, 255)]  # Swift systemBlue (RGB)
        
        print(f"Loading model from {weights_path}...")
        # 使用 ultralytics 的 hub 加载方式，或者直接加载本地 custom 模型
        # 这里假设用户有 yolov5 仓库或安装了 ultralytics 包
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
            self.model.conf = conf_thres
            self.model.iou = iou_thres
            # self.model.classes = [0] # 如果你的模型只有 figure 一类，通常是 class 0
            self.model.to(self.device)
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def detect(self, image: Image.Image) -> list[Detection]:
        """
        执行推理。
        YOLOv5 的 PyTorch 接口自动处理 Letterbox 和 Normalization。
        """
        if self.model is None:
            return []

        # 推理
        results = self.model(image, size=640)
        
        # 解析结果 pandas 格式方便处理
        df = results.pandas().xyxy[0] 
        
        detections = []
        for _, row in df.iterrows():
            # 过滤类别 (如果模型有多类)
            if row['name'] not in self.class_labels and len(self.class_labels) > 0:
                 # 如果模型训练时类名不对，可能需要按 row['class'] 索引判断
                 pass 

            # 坐标
            box = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
            
            # 颜色
            color = self.colors[0] # 默认蓝色
            
            detections.append(Detection(
                box=box,
                confidence=float(row['confidence']),
                label=row['name'],
                color=color
            ))
            
        return detections