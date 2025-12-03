from PIL import Image, ImageDraw, ImageFont
from .models import Detection
import numpy as np

class Visualizer:
    
    @staticmethod
    def draw_annotations(image: Image.Image, detections: list[Detection]) -> Image.Image:
        """
        对应 Swift: drawAnnotationsOnUIImage
        在图片上画框和标签
        """
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)
        
        # 字体设置 (尽量找个存在的字体，否则用默认)
        try:
            font = ImageFont.truetype("arial.ttf", size=max(12, int(image.width * 0.015)))
        except:
            font = ImageFont.load_default()

        line_width = max(2, int(min(image.width, image.height) * 0.004))

        for det in detections:
            # 画框
            draw.rectangle(det.box, outline=det.color, width=line_width)
            
            # 画标签
            if det.label:
                label_text = f"{det.label} : {int(det.confidence * 100)}"
                # 文本背景
                text_bbox = draw.textbbox((det.box[0], det.box[1]), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # 调整文字位置
                text_x = det.box[0] + line_width
                text_y = max(det.box[1] - text_height - 5, 0)
                
                draw.text((text_x, text_y), label_text, fill=det.color, font=font, stroke_width=1)

        return draw_img

    @staticmethod
    def crop_detections(image: Image.Image, detections: list[Detection]) -> list[Image.Image]:
        """
        对应 Swift: cropDetectionsHD
        """
        crops = []
        for det in detections:
            # PIL crop 需要 (left, upper, right, lower)
            # 确保坐标不越界
            x1, y1, x2, y2 = det.box
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(image.width, x2); y2 = min(image.height, y2)
            
            if x2 > x1 and y2 > y1:
                crop = image.crop((x1, y1, x2, y2))
                crops.append(crop)
        return crops