from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import os
import time
from collections import deque
import json
from loguru import logger

class FaceRecognizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_path = os.path.join('./models', 'buffalo_l')
        
        # 检查模型文件是否存在
        # if not os.path.exists(model_path):
        #     print("正在下载人脸识别模型...")
        #     self.app = FaceAnalysis(
        #         name='buffalo_l',
        #         root='./models',
        #         providers=['CPUExecutionProvider']
        #     )
        # else:
        print("使用已下载的人脸识别模型...")
        self.app = FaceAnalysis(
            name='buffalo_l',
            root='./models',
            providers=['CPUExecutionProvider'],
            download=False
        )
            
        # 使用更小的检测尺寸以提高性能
        self.app.prepare(ctx_id=-1, det_size=(256, 256))  # 进一步降低检测尺寸
        self.known_faces: Dict[str, np.ndarray] = {}
        self.last_detection_time = 0
        self.detection_interval = 1.0  # 基础检测间隔
        self.last_results = None
        self.skip_frames = 4  # 基础帧跳过数量
        self.frame_count = 0
        
        # 添加结果缓存
        self.result_cache = deque(maxlen=10)  # 缓存最近10次的结果
        self.cache_duration = 2.0  # 缓存有效期（秒）
        
        # 性能监控
        self.processing_times = deque(maxlen=30)  # 记录最近30次处理时间
        self.last_face_detected = False
        self.adaptive_skip_frames = self.skip_frames
        
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """检测人脸"""
        start_time = time.time()
        current_time = time.time()
        self.frame_count += 1
        
        # 自适应处理频率
        if self.last_face_detected:
            # 如果上一帧检测到人脸，减少跳过帧数
            self.adaptive_skip_frames = max(2, self.adaptive_skip_frames - 1)
        else:
            # 如果上一帧没有检测到人脸，增加跳过帧数
            self.adaptive_skip_frames = min(8, self.adaptive_skip_frames + 1)
        
        # 检查缓存
        if self.result_cache:
            last_result, last_time = self.result_cache[-1]
            if current_time - last_time < self.cache_duration:
                return last_result
        
        # 如果距离上次检测时间太短，返回缓存结果
        if current_time - self.last_detection_time < self.detection_interval and self.last_results is not None:
            return self.last_results
            
        # 每隔几帧处理一次
        if self.frame_count % self.adaptive_skip_frames != 0:
            return self.last_results if self.last_results is not None else {
                "detected": False,
                "face_embedding": None,
                "confidence": 0.0,
                "bbox": None,
                "pose": None
            }
            
        try:
            # 缩小图像尺寸以提高性能
            height, width = frame.shape[:2]
            scale = min(1.0, 512 / max(width, height))  # 降低最大尺寸
            if scale < 1.0:
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            
            # 检测人脸
            faces = self.app.get(frame)
            
            if len(faces) == 0:
                result = {
                    "detected": False,
                    "face_embedding": None,
                    "confidence": 0.0,
                    "bbox": None,
                    "pose": None
                }
                self.last_face_detected = False
            else:
                # 获取最大的人脸
                face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
                
                # 检查置信度
                if face.det_score < self.config.get("face_confidence_threshold", 0.6):
                    result = {
                        "detected": False,
                        "face_embedding": None,
                        "confidence": face.det_score,
                        "bbox": None,
                        "pose": None
                    }
                    self.last_face_detected = False
                else:
                    # 获取人脸特征
                    embedding = face.embedding
                    # 转换bbox坐标回原始图像尺寸
                    bbox = face.bbox.copy()
                    if scale < 1.0:
                        bbox = bbox / scale
                        
                    # 计算姿态角度
                    pose = self._calculate_pose(face)
                    
                    result = {
                        "detected": True,
                        "face_embedding": embedding.tolist(),
                        "confidence": face.det_score,
                        "bbox": bbox.tolist(),
                        "pose": pose
                    }
                    self.last_face_detected = True
            
            # 更新缓存
            self.last_results = result
            self.result_cache.append((result, current_time))
            self.last_detection_time = current_time
            
            # 记录处理时间
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # 如果处理时间过长，增加跳过帧数
            if processing_time > 0.1:  # 如果处理时间超过100ms
                self.adaptive_skip_frames = min(8, self.adaptive_skip_frames + 1)
            
            return result
            
        except Exception as e:
            print(f"人脸检测失败: {e}")
            result = {
                "detected": False,
                "face_embedding": None,
                "confidence": 0.0,
                "bbox": None,
                "pose": None
            }
            self.result_cache.append((result, current_time))
            self.last_face_detected = False
            return result
            
    def _calculate_pose(self, face) -> Dict[str, float]:
        """计算人脸姿态角度"""
        try:
            # 获取人脸关键点
            landmarks = face.kps
            
            # 计算姿态角度
            # 使用关键点计算欧拉角
            # 这里使用简化的计算方法，实际应用中可能需要更复杂的3D重建
            if landmarks is not None and len(landmarks) >= 5:
                # 计算左右眼中心点
                left_eye = np.mean(landmarks[0:2], axis=0)
                right_eye = np.mean(landmarks[2:4], axis=0)
                
                # 计算眼睛连线的角度（偏航角）
                eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                yaw = np.degrees(eye_angle)
                
                # 计算眼睛到嘴巴的垂直距离（俯仰角）
                mouth = landmarks[4]
                eye_center = (left_eye + right_eye) / 2
                vertical_angle = np.arctan2(mouth[1] - eye_center[1], mouth[0] - eye_center[0])
                pitch = np.degrees(vertical_angle)
                
                # 计算眼睛连线的倾斜角度（翻滚角）
                roll = np.degrees(eye_angle)
                
                # 使用insightface的姿态估计
                if hasattr(face, 'pose'):
                    # 如果insightface提供了姿态数据，使用它
                    yaw = face.pose[0]
                    pitch = face.pose[1]
                    roll = face.pose[2]
                
                return {
                    "yaw": float(yaw),
                    "pitch": float(pitch),
                    "roll": float(roll)
                }
            else:
                return {
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0
                }
        except Exception as e:
            print(f"姿态计算失败: {e}")
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0
            }
        
    def register_face(self, user_id: str, face_embedding: np.ndarray):
        """注册人脸"""
        self.known_faces[user_id] = face_embedding
        
    def match_face(self, face_embedding: np.ndarray, threshold: float = 0.6) -> Optional[str]:
        """匹配人脸"""
        if not self.known_faces:
            return None
            
        # 计算相似度
        similarities = {}
        for user_id, known_embedding in self.known_faces.items():
            similarity = np.dot(face_embedding, known_embedding) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
            )
            similarities[user_id] = similarity
            
        # 找到最相似的用户
        best_match = max(similarities.items(), key=lambda x: x[1])
        if best_match[1] >= threshold:
            return best_match[0]
            
        return None
        
    def draw_face(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """在图像上绘制人脸框"""
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame 

    def save_known_faces(self, save_path: str = "models/known_faces.json"):
        """保存已知人脸特征到本地"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump({k: v.tolist() for k, v in self.known_faces.items()}, f)
            logger.info(f"已知人脸特征已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存已知人脸特征失败: {e}")

    def load_known_faces(self, load_path: str = "models/known_faces.json"):
        """从本地加载已知人脸特征"""
        try:
            if os.path.exists(load_path):
                with open(load_path, 'r') as f:
                    data = json.load(f)
                    self.known_faces = {k: np.array(v) for k, v in data.items()}
                logger.info(f"已知人脸特征已加载: {load_path}")
        except Exception as e:
            logger.error(f"加载已知人脸特征失败: {e}")

    def reset(self):
        """重置检测缓存和状态"""
        self.last_results = None
        self.result_cache.clear()
        self.last_face_detected = False
        self.frame_count = 0
        logger.info("人脸识别器已重置")

    def clear_known_faces(self):
        """清空已知人脸特征"""
        self.known_faces.clear()
        logger.info("已知人脸特征已清空")

    def close(self):
        """资源释放"""
        self.reset()
        self.clear_known_faces()
        logger.info("人脸识别器已关闭") 