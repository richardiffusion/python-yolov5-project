from pdf2image import convert_from_path, convert_from_bytes
import numpy as np
from PIL import Image

class PDFProcessor:
    @staticmethod
    def pdf_to_images(file_bytes, dpi=200):
        """
        将 PDF 字节流转换为 PIL Images 列表。
        Swift 中使用了 scale 策略，这里我们用 DPI 控制分辨率。
        DPI 200 大约对应 Swift 中的 detectScale 2.0 (视具体屏幕而定，但在服务端够用了)
        """
        try:
            # convert_from_bytes 需要系统安装 poppler
            images = convert_from_bytes(file_bytes, dpi=dpi, fmt='jpeg')
            return images
        except Exception as e:
            print(f"Error converting PDF: {e}")
            return []

    @staticmethod
    def get_hd_image(pil_image, target_long_side=3200):
        """对应 Swift 的 renderPDFPageToHD"""
        w, h = pil_image.size
        ratio = target_long_side / max(w, h)
        if ratio < 1.0: # 只有比目标大才缩小，否则保持原样
            new_size = (int(w * ratio), int(h * ratio))
            return pil_image.resize(new_size, Image.Resampling.LANCZOS)
        return pil_image