> 本项目是 https://github.com/TinyCircl/CoreML-YOLOv5-ForPaper 的Python重写版本，将Swift实现转换为Python/PyTorch实现。
> 感谢原作者 [@TinyCircl.AI] [@xuao575] 的优秀工作。


##  我的实现
- **仓库地址**：[https://github.com/richardiffusion/python-yolov5-project]
- **语言/框架**：Python + PyTorch/YOLOv5
- **保持的核心功能**：PDF解析、文本检测、OCR处理、可视化
- **架构变化**：参考了原项目的模块设计，但用Python包结构重构

##  重写目的
1. 学习项目架构设计思路
2. 为需要跨平台（Windows/Linux）使用的开发者提供Python版本参考

## 文件结构
```txt
pdf_detector_py/
├── main.py                  # [入口] 程序的启动文件，包含 UI 界面代码 (Streamlit)
├── requirements.txt         # [配置] 依赖库列表 (torch, streamlit, opencv 等)
├── weights/                 # [资源] 存放模型权重文件
│   └── yolov5s.pt           # <--- ⚠️ 注意：这里放入 .pt 模型
└── core/                    # [逻辑包] 存放所有的后端处理逻辑
    ├── __init__.py          
    ├── detector.py          # [核心] 负责加载模型、执行 YOLOv5 推理
    ├── pdf_utils.py         # [工具] 负责将 PDF 转换为图片
    ├── ocr_engine.py        # [工具] 负责 OCR 文字识别
    ├── visualizer.py        # [工具] 负责在图片上画框、裁剪图片
    └── models.py            # [数据] 定义 Detection, PageResult 等数据结构
```

## 依赖
streamlit
torch
torchvision
opencv-python-headless
numpy
pdf2image
pytesseract
Pillow
pandas
ultralytics

## 运行
pip install -r requirements.txt

streamlit run main.py