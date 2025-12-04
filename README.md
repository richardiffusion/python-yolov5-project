> 本项目是 https://github.com/TinyCircl/CoreML-YOLOv5-ForPaper 的Python重写版本，将Swift实现转换为Python/PyTorch实现。
> 感谢原作者 [@TinyCircl.AI] [@xuao575] 的优秀工作。

# YOLOv5 PDF Figure Detector (Python Version)
这是一个基于 Python Streamlit 的 YOLOv5 推理应用，重写自原 iOS (CoreML) 项目。
它能够读取 PDF 文件，将其转换为图像，识别其中的插图（Figure），并使用 OCR 提取相关的文字说明。

## 📂 项目结构
```text
.
├── main.py                  # [入口] Streamlit 启动文件
├── requirements.txt         # Python 依赖库
├── weights/                 # 存放模型权重
│   └── best.pt              # 请在此放入训练好的 YOLOv5 模型
├── packages/                # 核心逻辑包
│   ├── detector.py          # YOLOv5 推理 (基于 ultralytics)
│   ├── pdf_processor.py     # PDF 转图片 (基于 pdf2image)
│   ├── ocr_engine.py        # OCR 识别 (基于 Tesseract)
│   ├── visualizer.py        # 绘图与裁剪工具
│   └── models.py            # 数据结构定义
└── venv/                    # (自动生成) Python 虚拟环境
```

## 🛠️ 环境准备 (Prerequisites)
本项目依赖两个外部系统工具，必须安装才能运行：

1. 安装 Poppler (用于 PDF 转图片)
Windows:

下载 Poppler for Windows。

解压并将 bin 文件夹路径（例如 F:\Program Files\Poppler\Library\bin）添加到系统 Path 环境变量。

Mac: brew install poppler

2. 安装 Tesseract-OCR (用于文字识别)
Windows:

下载并安装 Tesseract-OCR。

将安装目录（例如 F:\Program Files\Tesseract-OCR）添加到系统 Path 环境变量。

重要：确保安装目录下 tessdata 文件夹内包含 eng.traineddata (英文) 和 chi_sim.traineddata (简体中文) 语言包。缺一不可。

如果缺失，请新建环境变量 TESSDATA_PREFIX 指向 tessdata 文件夹路径。

Mac: brew install tesseract tesseract-lang

## 🚀 快速开始
1. 创建虚拟环境
建议使用虚拟环境以隔离依赖：
```Bash
# 创建
python -m venv venv

# 激活 (Windows)
.\venv\Scripts\activate

# 激活 (Mac/Linux)
source venv/bin/activate
```
2. 安装 Python 依赖
```Bash
pip install -r requirements.txt
```
3. 准备模型
将你训练好的 .pt 权重文件命名为 best.pt，放入 weights/ 文件夹中。

4. 运行应用
```Bash
streamlit run main.py
```

## ⚠️ 常见问题
Q: 报错 TesseractNotFoundError 或 tesseract is not installed? 
A: 请检查是否将 Tesseract 的安装目录添加到了系统 Path 环境变量中。如果添加后无效，请重启终端或电脑。

Q: 报错 Error opening data file ... chi_sim.traineddata? 
A: tessdata 文件夹缺少对应的语言包。请去 GitHub 下载 chi_sim.traineddata 和 eng.traineddata 并放入该文件夹。

Q: 报错 Unable to get page count? 
A: Poppler 未正确安装或未配置 Path 环境变量。