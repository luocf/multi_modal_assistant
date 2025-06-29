import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Deque
from collections import deque
import time
from loguru import logger
import mediapipe as mp

class LipFeatureExtractor(nn.Module):
    """唇部特征提取网络"""
    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(96 * 5 * 5, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class AudioVisualSyncNet(nn.Module):
    """音视频同步性分析网络"""
    def __init__(self, audio_dim: int = 256, visual_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        self.audio_fc = nn.Linear(audio_dim, hidden_dim)
        self.visual_fc = nn.Linear(visual_dim, hidden_dim)
        
        self.sync_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, audio_feat: torch.Tensor, visual_feat: torch.Tensor) -> torch.Tensor:
        audio_hidden = F.relu(self.audio_fc(audio_feat))
        visual_hidden = F.relu(self.visual_fc(visual_feat))
        
        combined = torch.cat([audio_hidden, visual_hidden], dim=-1)
        sync_score = self.sync_fc(combined)
        
        return sync_score

@dataclass
class LipFrame:
    """唇部帧数据"""
    landmarks: np.ndarray  # 唇部关键点
    roi: np.ndarray  # 唇部区域图像
    timestamp: float  # 时间戳
    motion: Optional[np.ndarray] = None  # 运动向量

class LipAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化唇部分析器
        Args:
            config: 配置字典，包含：
                - lip_roi_size: 唇部ROI大小
                - sync_window_size: 同步性分析窗口大小
                - min_sync_score: 最小同步性分数
                - feature_dim: 特征维度
        """
        self.config = config
        self.roi_size = config.get("lip_roi_size", (40, 40))
        self.sync_window_size = config.get("sync_window_size", 16)
        self.min_sync_score = config.get("min_sync_score", 0.7)
        self.feature_dim = config.get("feature_dim", 256)
        
        # 初始化MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 唇部关键点索引
        self.lip_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            95, 78, 191, 80, 81, 82
        ]
        
        # 初始化特征提取器和同步性分析网络
        self.lip_feature_extractor = LipFeatureExtractor(output_dim=self.feature_dim)
        self.sync_net = AudioVisualSyncNet(
            audio_dim=self.feature_dim,
            visual_dim=self.feature_dim
        )
        
        # 加载预训练模型（如果有）
        self._load_models()
        
        # 数据缓冲
        self.lip_frames: Deque[LipFrame] = deque(maxlen=self.sync_window_size)
        self.audio_features: Deque[Tuple[np.ndarray, float]] = deque(maxlen=self.sync_window_size)
        
        logger.info("唇部分析器初始化完成")
        
    def _load_models(self):
        """加载预训练模型"""
        try:
            # 这里应该加载实际的预训练模型
            # 目前使用随机初始化的模型作为示例
            pass
        except Exception as e:
            logger.error(f"加载预训练模型失败: {e}")
            
    def extract_lip_roi(self, frame: np.ndarray, face_landmarks: np.ndarray) -> Optional[np.ndarray]:
        """提取唇部ROI区域"""
        try:
            # 获取唇部关键点
            lip_points = face_landmarks[self.lip_indices]
            
            # 计算边界框
            min_x, min_y = np.min(lip_points, axis=0)
            max_x, max_y = np.max(lip_points, axis=0)
            
            # 添加边距
            margin = 10
            min_x = max(0, int(min_x - margin))
            min_y = max(0, int(min_y - margin))
            max_x = min(frame.shape[1], int(max_x + margin))
            max_y = min(frame.shape[0], int(max_y + margin))
            
            # 提取ROI
            roi = frame[min_y:max_y, min_x:max_x]
            
            # 调整大小
            roi = cv2.resize(roi, self.roi_size)
            
            return roi
            
        except Exception as e:
            logger.error(f"提取唇部ROI失败: {e}")
            return None
            
    def compute_lip_motion(self, current_landmarks: np.ndarray,
                          previous_landmarks: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """计算唇部运动"""
        if previous_landmarks is None:
            return None
            
        try:
            # 计算关键点位移
            motion = current_landmarks - previous_landmarks
            
            # 计算光流
            # 这里可以使用更复杂的光流算法
            return motion
            
        except Exception as e:
            logger.error(f"计算唇部运动失败: {e}")
            return None
            
    def extract_lip_features(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """提取唇部特征"""
        try:
            # 预处理图像
            roi_tensor = torch.from_numpy(roi).float()
            roi_tensor = roi_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # 提取特征
            with torch.no_grad():
                features = self.lip_feature_extractor(roi_tensor)
                
            return features.numpy()
            
        except Exception as e:
            logger.error(f"提取唇部特征失败: {e}")
            return None
            
    def analyze_frame(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """分析视频帧"""
        try:
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测面部关键点
            results = self.face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                return {
                    "success": False,
                    "message": "未检测到人脸"
                }
                
            # 获取第一个人脸的关键点
            face_landmarks = np.array([[point.x * frame.shape[1], point.y * frame.shape[0]]
                                     for point in results.multi_face_landmarks[0].landmark])
            
            # 提取唇部ROI
            roi = self.extract_lip_roi(frame, face_landmarks)
            if roi is None:
                return {
                    "success": False,
                    "message": "提取唇部ROI失败"
                }
                
            # 计算唇部运动
            previous_frame = self.lip_frames[-1] if self.lip_frames else None
            motion = self.compute_lip_motion(
                face_landmarks[self.lip_indices],
                previous_frame.landmarks if previous_frame else None
            )
            
            # 创建新的帧数据
            lip_frame = LipFrame(
                landmarks=face_landmarks[self.lip_indices],
                roi=roi,
                timestamp=timestamp,
                motion=motion
            )
            self.lip_frames.append(lip_frame)
            
            return {
                "success": True,
                "lip_frame": lip_frame
            }
            
        except Exception as e:
            logger.error(f"分析视频帧失败: {e}")
            return {
                "success": False,
                "message": str(e)
            }
            
    def analyze_audio_sync(self, audio_feature: np.ndarray,
                          timestamp: float) -> Dict[str, Any]:
        """分析音视频同步性"""
        try:
            # 添加音频特征到缓冲区
            self.audio_features.append((audio_feature, timestamp))
            
            if len(self.lip_frames) < 2 or len(self.audio_features) < 2:
                return {
                    "sync_score": 0.0,
                    "message": "数据不足"
                }
                
            # 获取最近的唇部帧
            recent_lip_frame = self.lip_frames[-1]
            
            # 提取唇部特征
            lip_features = self.extract_lip_features(recent_lip_frame.roi)
            if lip_features is None:
                return {
                    "sync_score": 0.0,
                    "message": "提取唇部特征失败"
                }
                
            # 转换为tensor
            lip_tensor = torch.from_numpy(lip_features).float()
            audio_tensor = torch.from_numpy(audio_feature).float()
            
            # 计算同步性分数
            with torch.no_grad():
                sync_score = self.sync_net(audio_tensor, lip_tensor)
                
            return {
                "sync_score": float(sync_score),
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"分析音视频同步性失败: {e}")
            return {
                "sync_score": 0.0,
                "message": str(e)
            }
            
    def get_sync_confidence(self, window_size: int = None) -> float:
        """获取一段时间内的同步性置信度"""
        if not self.lip_frames or not self.audio_features:
            return 0.0
            
        try:
            # 确定分析窗口大小
            if window_size is None:
                window_size = len(self.lip_frames)
            else:
                window_size = min(window_size, len(self.lip_frames))
                
            # 获取最近的帧
            recent_frames = list(self.lip_frames)[-window_size:]
            recent_audio = list(self.audio_features)[-window_size:]
            
            sync_scores = []
            for frame, (audio, _) in zip(recent_frames, recent_audio):
                # 提取特征
                lip_features = self.extract_lip_features(frame.roi)
                if lip_features is None:
                    continue
                    
                # 计算同步性分数
                lip_tensor = torch.from_numpy(lip_features).float()
                audio_tensor = torch.from_numpy(audio).float()
                
                with torch.no_grad():
                    sync_score = self.sync_net(audio_tensor, lip_tensor)
                    sync_scores.append(float(sync_score))
                    
            if not sync_scores:
                return 0.0
                
            # 计算平均同步性分数
            return np.mean(sync_scores)
            
        except Exception as e:
            logger.error(f"计算同步性置信度失败: {e}")
            return 0.0
            
    def reset(self):
        """重置分析器状态"""
        self.lip_frames.clear()
        self.audio_features.clear()
        
    def draw_lip_landmarks(self, frame: np.ndarray, lip_frame: LipFrame) -> np.ndarray:
        """绘制唇部关键点和ROI"""
        try:
            # 绘制关键点
            for point in lip_frame.landmarks:
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
                
            # 绘制ROI边界框
            min_x, min_y = np.min(lip_frame.landmarks, axis=0)
            max_x, max_y = np.max(lip_frame.landmarks, axis=0)
            cv2.rectangle(frame,
                         (int(min_x), int(min_y)),
                         (int(max_x), int(max_y)),
                         (0, 255, 0), 2)
                         
            return frame
            
        except Exception as e:
            logger.error(f"绘制唇部关键点失败: {e}")
            return frame 