import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, List, Optional, Tuple
import time
from collections import deque
from enum import Enum
from loguru import logger

class GestureType(Enum):
    """手势类型"""
    UNKNOWN = "unknown"
    STOP = "stop"  # 停止手势
    CONFIRM = "confirm"  # 确认手势（竖起大拇指）
    CANCEL = "cancel"  # 取消手势（摇手）
    POINT = "point"  # 指向手势
    WAVE = "wave"  # 挥手手势
    PALM = "palm"  # 手掌手势

class GestureRecognizer:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化手势识别器
        Args:
            config: 配置字典，包含：
                - gesture_threshold: 手势识别阈值
                - gesture_history_size: 手势历史大小
                - detection_interval: 检测间隔（秒）
        """
        self.config = config
        self.gesture_threshold = config.get("gesture_threshold", 0.7)
        self.history_size = config.get("gesture_history_size", 10)
        self.detection_interval = config.get("detection_interval", 0.1)
        
        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化手部检测器
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 手势历史记录
        self.gesture_history = deque(maxlen=self.history_size)
        self.last_detection_time = 0
        
        # 动作状态追踪
        self.action_states = {
            "wave": {
                "positions": deque(maxlen=5),  # 存储最近5帧的手部位置
                "last_direction": None,  # 上一次移动方向
                "direction_changes": 0  # 方向改变次数
            }
        }
        
        logger.info("手势识别器初始化完成")
        
    def detect_gesture(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        检测手势
        Returns:
            {
                "gesture": GestureType,
                "confidence": float,
                "hand_landmarks": List[Dict],
                "hand_world_landmarks": List[Dict],
                "handedness": List[str]
            }
        """
        current_time = time.time()
        
        # 如果距离上次检测时间太短，返回最近的结果
        if current_time - self.last_detection_time < self.detection_interval and self.gesture_history:
            return self.gesture_history[-1]
            
        try:
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if not results.multi_hand_landmarks:
                result = {
                    "gesture": GestureType.UNKNOWN,
                    "confidence": 0.0,
                    "hand_landmarks": None,
                    "hand_world_landmarks": None,
                    "handedness": []
                }
            else:
                # 获取手势类型和置信度
                gesture, confidence = self._classify_gesture(
                    results.multi_hand_landmarks,
                    results.multi_hand_world_landmarks,
                    results.multi_handedness
                )
                
                result = {
                    "gesture": gesture,
                    "confidence": confidence,
                    "hand_landmarks": [
                        self._landmarks_to_dict(landmarks)
                        for landmarks in results.multi_hand_landmarks
                    ],
                    "hand_world_landmarks": [
                        self._landmarks_to_dict(landmarks)
                        for landmarks in results.multi_hand_world_landmarks
                    ] if results.multi_hand_world_landmarks else None,
                    "handedness": [
                        hand.classification[0].label
                        for hand in results.multi_handedness
                    ]
                }
                
            # 更新历史记录
            self.gesture_history.append(result)
            self.last_detection_time = current_time
            
            return result
            
        except Exception as e:
            logger.error(f"手势检测失败: {e}")
            return {
                "gesture": GestureType.UNKNOWN,
                "confidence": 0.0,
                "hand_landmarks": None,
                "hand_world_landmarks": None,
                "handedness": []
            }
            
    def _landmarks_to_dict(self, landmarks) -> List[Dict[str, float]]:
        """将landmarks转换为字典格式"""
        return [
            {"x": point.x, "y": point.y, "z": point.z}
            for point in landmarks.landmark
        ]
        
    def _classify_gesture(
        self,
        hand_landmarks: List[Any],
        world_landmarks: List[Any],
        handedness: List[Any]
    ) -> Tuple[GestureType, float]:
        """
        分类手势类型
        Returns:
            (gesture_type, confidence)
        """
        if not hand_landmarks:
            return GestureType.UNKNOWN, 0.0
            
        # 获取第一只手的landmarks
        landmarks = hand_landmarks[0].landmark
        
        # 计算关键点之间的角度和距离
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # 检测停止手势（手掌打开）
        if self._is_palm_open(landmarks):
            # 更新手部位置用于检测挥手
            self._update_wave_detection(wrist)
            # 如果检测到挥手动作
            if self._detect_wave_gesture():
                return GestureType.WAVE, 0.8
            return GestureType.STOP, 0.9
            
        # 检测确认手势（竖起大拇指）
        if self._is_thumb_up(thumb_tip, thumb_ip, wrist):
            return GestureType.CONFIRM, 0.85
            
        # 检测指向手势（食指指向）
        if self._is_pointing(index_tip, index_pip, middle_tip):
            return GestureType.POINT, 0.8
            
        # 检测手掌手势
        if self._is_palm_facing_camera(landmarks):
            return GestureType.PALM, 0.85
            
        return GestureType.UNKNOWN, 0.5
        
    def _is_palm_open(self, landmarks) -> bool:
        """检测手掌是否打开"""
        # 检查所有手指是否伸直
        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_pips = [
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        # 检查每个手指是否伸直
        fingers_extended = []
        for tip, pip in zip(finger_tips, finger_pips):
            tip_pos = landmarks[tip]
            pip_pos = landmarks[pip]
            # 如果指尖高于关节，则认为手指伸直
            fingers_extended.append(tip_pos.y < pip_pos.y)
            
        # 如果大多数手指伸直，则认为手掌打开
        return sum(fingers_extended) >= 4
        
    def _is_thumb_up(self, thumb_tip, thumb_ip, wrist) -> bool:
        """检测是否为竖起大拇指手势"""
        # 检查拇指是否高于手腕且指向上方
        return (thumb_tip.y < wrist.y and
                thumb_tip.y < thumb_ip.y and
                abs(thumb_tip.x - thumb_ip.x) < 0.1)
                
    def _is_pointing(self, index_tip, index_pip, middle_tip) -> bool:
        """检测是否为指向手势"""
        # 检查食指是否伸直而其他手指弯曲
        return (index_tip.y < index_pip.y and
                middle_tip.y > index_pip.y)
                
    def _is_palm_facing_camera(self, landmarks) -> bool:
        """检测手掌是否朝向摄像头"""
        # 使用手指关节的相对位置来判断手掌方向
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # 计算手掌法向量
        palm_normal = np.array([
            middle_mcp.x - wrist.x,
            middle_mcp.y - wrist.y,
            middle_mcp.z - wrist.z
        ])
        
        # 如果z分量较大，说明手掌朝向摄像头
        return abs(palm_normal[2]) > 0.1
        
    def _update_wave_detection(self, wrist):
        """更新挥手检测状态"""
        self.action_states["wave"]["positions"].append((wrist.x, wrist.y))
        
        if len(self.action_states["wave"]["positions"]) >= 2:
            current_pos = self.action_states["wave"]["positions"][-1]
            prev_pos = self.action_states["wave"]["positions"][-2]
            
            # 计算移动方向
            current_direction = "right" if current_pos[0] > prev_pos[0] else "left"
            
            # 检查方向是否改变
            if (self.action_states["wave"]["last_direction"] and
                current_direction != self.action_states["wave"]["last_direction"]):
                self.action_states["wave"]["direction_changes"] += 1
                
            self.action_states["wave"]["last_direction"] = current_direction
            
    def _detect_wave_gesture(self) -> bool:
        """检测是否完成挥手手势"""
        # 如果方向改变次数达到阈值，认为是挥手手势
        if self.action_states["wave"]["direction_changes"] >= 2:
            # 重置状态
            self.action_states["wave"]["direction_changes"] = 0
            self.action_states["wave"]["last_direction"] = None
            self.action_states["wave"]["positions"].clear()
            return True
        return False
        
    def draw_landmarks(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """在图像上绘制手部关键点"""
        if not result["hand_landmarks"]:
            return frame
            
        try:
            # 转换回mediapipe格式
            for landmarks_dict in result["hand_landmarks"]:
                landmarks = self.mp_hands.HandLandmark()
                for i, point in enumerate(landmarks_dict):
                    landmark = landmarks.landmark.add()
                    landmark.x = point["x"]
                    landmark.y = point["y"]
                    landmark.z = point["z"]
                    
                # 绘制关键点和连接线
                self.mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
            # 绘制手势类型
            gesture = result["gesture"]
            confidence = result["confidence"]
            cv2.putText(
                frame,
                f"{gesture.value}: {confidence:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"绘制手部关键点失败: {e}")
            return frame
            
    def reset_history(self):
        """重置手势历史记录"""
        self.gesture_history.clear()
        self.last_detection_time = 0
        self.action_states["wave"]["positions"].clear()
        self.action_states["wave"]["last_direction"] = None
        self.action_states["wave"]["direction_changes"] = 0 