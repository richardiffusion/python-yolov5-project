from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image
import numpy as np

@dataclass
class Detection:
    """对应 Swift 的 Detection 结构体"""
    box: List[int] # [x1, y1, x2, y2] 像素坐标
    confidence: float
    label: Optional[str]
    color: tuple # (R, G, B)

@dataclass
class PageResult:
    """对应 Swift 的 PageResult 结构体"""
    page_index: int
    annotated_image: Image.Image
    crops: List[Image.Image]
    captions: List[str] = field(default_factory=list)