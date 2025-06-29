import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import time
from loguru import logger
import torch
import torchvision.transforms as transforms
from collections import deque

class VideoEnhancer:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化视频增强器
        Args:
            config: 配置字典，包含：
                - target_fps: 目标帧率
                - min_face_size: 最小人脸尺寸
                - quality_threshold: 图像质量阈值
                - use_gpu: 是否使用GPU
        """
        self.config = config
        self.target_fps = config.get("target_fps", 30)
        self.min_face_size = config.get("min_face_size", 30)
        self.quality_threshold = config.get("quality_threshold", 0.5)
        
        # 初始化人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 初始化图像增强模型
        self.enhancer = None
        if config.get("use_gpu", True) and torch.cuda.is_available():
            try:
                # 这里可以加载预训练的图像增强模型
                # 比如超分辨率或去噪模型
                pass
            except Exception as e:
                logger.warning(f"加载图像增强模型失败: {e}")
                
        # 性能统计
        self.stats = {
            "fps_history": deque(maxlen=30),
            "face_detect_times": deque(maxlen=30),
            "quality_scores": deque(maxlen=30),
            "last_frame_time": time.time(),
            "frame_count": 0
        }
        
        # 自适应参数
        self.adaptive_params = {
            "brightness": 1.0,
            "contrast": 1.0,
            "blur_kernel": 1,
            "current_fps": self.target_fps
        }
        
        logger.info("视频增强器初始化完成")
        
    def enhance(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """增强视频帧"""
        try:
            # 1. 基本验证
            if image is None or image.size == 0:
                return None, {"error": "无效的输入图像"}
            
            # 检查通道数
            if len(image.shape) != 3 or image.shape[2] != 3:
                return None, {"error": "图像必须是3通道RGB格式"}
            
            # 2. 质量评估
            quality_score = self._calculate_quality(image)
            if quality_score < self.quality_threshold:
                logger.warning(f"图像质量较低: {quality_score:.2f}")
            
            # 3. 自适应参数调整
            self._adapt_parameters(image)
            
            # 4. 人脸检测和处理
            faces = self._detect_faces(image)
            if faces:
                for (x, y, w, h) in faces:
                    # 扩大ROI区域
                    x = max(0, x - int(0.1 * w))
                    y = max(0, y - int(0.1 * h))
                    w = min(image.shape[1] - x, int(1.2 * w))
                    h = min(image.shape[0] - y, int(1.2 * h))
                    
                    # 对人脸区域进行特殊增强
                    face_roi = image[y:y+h, x:x+w]
                    enhanced_roi = self._enhance_face(face_roi)
                    image[y:y+h, x:x+w] = enhanced_roi
            
            # 5. 全局增强
            enhanced = self._enhance_global(image)
            
            # 6. 返回结果
            return enhanced, {
                "quality_score": quality_score,
                "num_faces": len(faces) if faces else 0,
                "enhanced": True
            }
            
        except Exception as e:
            logger.error(f"视频增强失败: {e}")
            return None, {"error": str(e)}

    def _update_stats(self, frame: np.ndarray):
        """更新性能统计"""
        try:
            current_time = time.time()
            
            # 更新FPS
            if self.stats["last_frame_time"] > 0:
                fps = 1.0 / (current_time - self.stats["last_frame_time"])
                self.stats["fps_history"].append(fps)
                
            self.stats["last_frame_time"] = current_time
            self.stats["frame_count"] += 1
            
            # 计算图像质量分数
            quality = self._calculate_quality(frame)
            self.stats["quality_scores"].append(quality)
            
            # 人脸检测时间统计
            face_detect_start = time.time()
            self._detect_faces(frame)
            self.stats["face_detect_times"].append(time.time() - face_detect_start)
            
        except Exception as e:
            logger.error(f"更新视频统计失败: {e}")
            
    def _adapt_parameters(self, image: np.ndarray) -> None:
        """自适应调整参数"""
        try:
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("图像必须是3通道RGB格式")
            
            # 1. 分析图像统计信息
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean_brightness = np.mean(gray)
            std_contrast = np.std(gray)
            
            # 2. 调整亮度范围
            if mean_brightness < 128:
                self.adaptive_params["brightness"] = 0.6
            else:
                self.adaptive_params["brightness"] = 0.4
            
            # 3. 调整对比度范围
            if std_contrast < 64:
                self.adaptive_params["contrast"] = 0.6
            else:
                self.adaptive_params["contrast"] = 0.4
            
        except Exception as e:
            logger.error(f"参数自适应调整失败: {e}")
            
    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测人脸"""
        try:
            # 1. 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 2. 人脸检测
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            # 3. 提取人脸区域
            face_regions = []
            for face in faces:
                x = face[0]
                y = face[1]
                w = face[2]
                h = face[3]
                
                # 检查人脸大小
                if w >= self.min_face_size and h >= self.min_face_size:
                    face_regions.append((x, y, w, h))
                
            return face_regions
            
        except Exception as e:
            logger.error(f"人脸检测失败: {e}")
            return []
            
    def _enhance_faces(self, frame: np.ndarray) -> np.ndarray:
        """增强人脸区域"""
        try:
            faces = self._detect_faces(frame)
            if len(faces) == 0:
                return frame
                
            enhanced = frame.copy()
            for (x, y, w, h) in faces:
                # 提取人脸ROI
                face_roi = enhanced[y:y+h, x:x+w]
                
                # 对人脸区域进行增强
                # 1. 局部对比度增强
                lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                enhanced_lab = cv2.merge((l,a,b))
                enhanced_face = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
                # 2. 边缘保持的平滑
                enhanced_face = cv2.edgePreservingFilter(
                    enhanced_face,
                    flags=1,
                    sigma_s=60,
                    sigma_r=0.4
                )
                
                # 将增强后的人脸放回原图
                enhanced[y:y+h, x:x+w] = enhanced_face
                
            return enhanced
            
        except Exception as e:
            logger.error(f"人脸增强失败: {e}")
            return frame
            
    def _calculate_quality(self, image: np.ndarray) -> float:
        """计算图像质量分数"""
        try:
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("图像必须是3通道RGB格式")
            
            # 1. 亮度评分
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            brightness_score = np.mean(gray) / 255.0
            
            # 2. 对比度评分
            contrast_score = np.std(gray) / 128.0
            
            # 3. 清晰度评分
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness_score = np.std(laplacian) / 128.0
            
            # 4. 噪声评分
            noise_score = 1.0 - (np.std(image) / 128.0)
            
            # 5. 综合评分
            quality_score = np.mean([
                brightness_score,
                contrast_score,
                sharpness_score,
                noise_score
            ])
            
            return float(quality_score)
            
        except Exception as e:
            logger.error(f"质量评估失败: {e}")
            return 0.0
            
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "fps_history": deque(maxlen=30),
            "face_detect_times": deque(maxlen=30),
            "quality_scores": deque(maxlen=30),
            "last_frame_time": time.time(),
            "frame_count": 0
        }
        self.adaptive_params = {
            "brightness": 1.0,
            "contrast": 1.0,
            "blur_kernel": 1,
            "current_fps": self.target_fps
        }

    def _enhance_global(self, image: np.ndarray) -> np.ndarray:
        """全局图像增强"""
        try:
            # 1. 亮度和对比度调整
            enhanced = cv2.convertScaleAbs(
                image,
                alpha=self.adaptive_params["contrast"],
                beta=self.adaptive_params["brightness"] * 50
            )
            
            # 2. 锐化
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # 3. 降噪
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced,
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21
            )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"全局增强失败: {e}")
            return image

    def _enhance_face(self, face_roi: np.ndarray) -> np.ndarray:
        """增强人脸区域"""
        try:
            # 1. 直方图均衡化
            lab = cv2.cvtColor(face_roi, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge((l,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # 2. 细节增强
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"人脸区域增强失败: {e}")
            return face_roi 