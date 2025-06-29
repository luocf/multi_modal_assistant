import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import cv2
from fer import FER
import time
from collections import deque
from loguru import logger
from deepface import DeepFace  # 新增导入

class EmotionRecognizer:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化表情识别器
        Args:
            config: 配置字典，包含：
                - emotion_threshold: 情绪识别阈值
                - emotion_history_size: 情绪历史大小
                - detection_interval: 检测间隔（秒）
                - model_type: 'fer' 或 'deepface'，选择底层模型
        """
        self.config = config
        self.emotion_threshold = config.get("emotion_threshold", 0.5)
        self.history_size = config.get("emotion_history_size", 10)
        self.detection_interval = config.get("detection_interval", 0.5)
        self.model_type = config.get("model_type", "fer")  # 新增模型类型
        
        # 初始化FER或DeepFace检测器
        if self.model_type == "fer":
            try:
                self.detector = FER(mtcnn=True)  # 使用MTCNN进行人脸检测
                logger.info("表情识别模型(FER)加载成功")
            except Exception as e:
                logger.error(f"表情识别模型(FER)加载失败: {e}")
                raise
        elif self.model_type == "deepface":
            self.detector = None  # DeepFace 无需提前初始化
            logger.info("表情识别模型(DeepFace)准备就绪")
        else:
            raise ValueError(f"不支持的表情识别模型类型: {self.model_type}")
        
        # 情绪历史记录
        self.emotion_history = deque(maxlen=self.history_size)
        self.last_detection_time = 0
        
        # 支持的情绪类型
        self.emotion_types = [
            "angry", "disgust", "fear", "happy",
            "sad", "surprise", "neutral"
        ]
        
    def detect_emotion(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        检测图像中的情绪
        Returns:
            {
                "emotion": str,  # 主要情绪
                "confidence": float,  # 置信度
                "all_emotions": Dict[str, float],  # 所有情绪的概率分布
                "face_box": List[int]  # 人脸框 [x, y, w, h]
            }
        """
        current_time = time.time()
        
        # 如果距离上次检测时间太短，返回最近的结果
        if current_time - self.last_detection_time < self.detection_interval and self.emotion_history:
            return self.emotion_history[-1]
            
        try:
            if self.model_type == "fer":
                # FER 检测逻辑
                results = self.detector.detect_emotions(frame)
                if not results:  # 没有检测到人脸
                    result = {
                        "emotion": "unknown",
                        "confidence": 0.0,
                        "all_emotions": {emotion: 0.0 for emotion in self.emotion_types},
                        "face_box": None
                    }
                else:
                    # 获取最大的人脸
                    face = max(results, key=lambda x: x["box"][2] * x["box"][3])
                    emotions = face["emotions"]
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                    result = {
                        "emotion": dominant_emotion[0],
                        "confidence": dominant_emotion[1],
                        "all_emotions": emotions,
                        "face_box": face["box"]
                    }
            elif self.model_type == "deepface":
                # DeepFace 检测逻辑
                deepface_result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                if isinstance(deepface_result, list):
                    deepface_result = deepface_result[0]
                emotions = deepface_result['emotion']
                dominant_emotion = deepface_result['dominant_emotion']
                confidence = emotions[dominant_emotion] / 100.0 if emotions[dominant_emotion] > 1 else emotions[dominant_emotion]
                result = {
                    "emotion": dominant_emotion,
                    "confidence": confidence,
                    "all_emotions": emotions,
                    "face_box": None  # DeepFace不返回人脸框
                }
            else:
                raise ValueError(f"不支持的表情识别模型类型: {self.model_type}")
            
            # 更新历史记录
            self.emotion_history.append(result)
            self.last_detection_time = current_time
            
            return result
            
        except Exception as e:
            logger.error(f"情绪检测失败: {e}")
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "all_emotions": {emotion: 0.0 for emotion in self.emotion_types},
                "face_box": None
            }
            
    def get_emotion_trend(self, window_size: int = None) -> Dict[str, float]:
        """
        获取情绪趋势
        Args:
            window_size: 时间窗口大小，默认使用全部历史
        Returns:
            各情绪的平均概率分布
        """
        if not self.emotion_history:
            return {emotion: 0.0 for emotion in self.emotion_types}
            
        # 确定使用的历史记录
        history = list(self.emotion_history)
        if window_size:
            history = history[-window_size:]
            
        # 计算平均情绪分布
        emotion_sums = {emotion: 0.0 for emotion in self.emotion_types}
        count = len(history)
        
        for record in history:
            emotions = record["all_emotions"]
            for emotion, prob in emotions.items():
                emotion_sums[emotion] += prob
                
        return {
            emotion: prob / count
            for emotion, prob in emotion_sums.items()
        }
        
    def get_dominant_emotion(self, window_size: int = None) -> Tuple[str, float]:
        """
        获取主导情绪
        Returns:
            (emotion, confidence)
        """
        trend = self.get_emotion_trend(window_size)
        dominant_emotion = max(trend.items(), key=lambda x: x[1])
        return dominant_emotion[0], dominant_emotion[1]
        
    def draw_emotion(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """在图像上绘制情绪标签"""
        if result["face_box"] is None:
            return frame
            
        try:
            x, y, w, h = result["face_box"]
            emotion = result["emotion"]
            confidence = result["confidence"]
            
            # 绘制人脸框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制情绪标签
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0), 2
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"绘制情绪标签失败: {e}")
            return frame
            
    def reset_history(self):
        """重置情绪历史记录"""
        self.emotion_history.clear()
        self.last_detection_time = 0 