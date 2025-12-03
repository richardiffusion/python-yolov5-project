import pytesseract
import re
import concurrent.futures
from PIL import Image

class OCREngine:
    
    # 对应 Swift 的 ocrLanguages: ["zh-Hans", "en"]
    # Tesseract 语言包通常是 'chi_sim+eng'
    LANG = 'chi_sim+eng' 

    @staticmethod
    def recognize_text(image: Image.Image) -> str:
        """同步 OCR 单张图片"""
        try:
            text = pytesseract.image_to_string(image, lang=OCREngine.LANG)
            return OCREngine.post_process(text)
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    @staticmethod
    def recognize_batch(images: list[Image.Image]) -> list[str]:
        """
        对应 Swift 的 ocrCaptions 并发处理。
        使用 ThreadPoolExecutor 模拟 TaskGroup
        """
        if not images:
            return []
        
        results = [""] * len(images)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_idx = {executor.submit(OCREngine.recognize_text, img): i for i, img in enumerate(images)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = ""
        return results

    @staticmethod
    def post_process(text: str) -> str:
        """
        对应 Swift 的 postProcessOCRText
        只保留 'Figure' 之后的内容，压缩空白
        """
        # 去首尾
        text = text.strip()
        
        # 查找 Figure (忽略大小写)
        match = re.search(r'Figure', text, re.IGNORECASE)
        if match:
            text = text[match.start():]
        
        # 删除所有换行 (Swift 选项 A)
        text = re.sub(r'[\r\n]+', '', text)
        
        # 压缩连续空格
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()