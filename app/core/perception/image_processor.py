from typing import Dict, Any, Optional
import cv2
import numpy as np
from loguru import logger

class ImageProcessor:
    def __init__(self):
        """初始化图像处理器"""
        # 初始化人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def process_image(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """处理图像数据"""
        try:
            # 将字节数据转换为numpy数组
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("无法解码图像数据")
                return None
                
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # 提取图像信息
            image_info = {
                "width": image.shape[1],
                "height": image.shape[0],
                "faces": []
            }
            
            # 处理检测到的人脸
            for (x, y, w, h) in faces:
                face_info = {
                    "position": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    }
                }
                image_info["faces"].append(face_info)
                
            return image_info
            
        except Exception as e:
            logger.error(f"处理图像失败: {e}")
            return None 

    def process_images(self, images_bytes: list) -> list:
        """批量处理图像数据"""
        results = []
        for image_bytes in images_bytes:
            result = self.process_image(image_bytes)
            results.append(result)
        self.logger.info(f"批量处理{len(images_bytes)}张图片")
        return results

    def reset(self):
        """重置图像处理器状态"""
        self.logger.info("图像处理器已重置")

    def close(self):
        """资源释放"""
        self.reset()
        self.logger.info("图像处理器已关闭") 