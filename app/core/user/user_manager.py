from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass, asdict
import time
from collections import deque
import json
import os
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from app.core.perception.voice_recognizer import VoiceRecognizer
from app.core.utils import compute_cosine_similarity
# 用户状态枚举
class UserState(Enum):
    UNKNOWN = "unknown"
    IDENTIFIED = "identified"
    ACTIVE = "active"
    INACTIVE = "inactive"

@dataclass
class UserProfile:
    """用户档案"""
    user_id: str
    name: str
    face_embeddings: List[np.ndarray]  # 存储多个面部特征
    last_seen: float
    visit_count: int = 0
    confidence_history: deque = None  # 记录最近N次的识别置信度
    state: UserState = UserState.UNKNOWN  # 用户状态
    last_state_change: float = 0  # 上次状态改变时间
    state_history: List[Dict] = None  # 状态历史记录
    
    def __post_init__(self):
        if self.confidence_history is None:
            self.confidence_history = deque(maxlen=10)  # 记录最近10次的置信度
        if self.state_history is None:
            self.state_history = []
            
    def add_confidence(self, confidence: float):
        """添加新的置信度记录"""
        self.confidence_history.append(confidence)
        
    def get_average_confidence(self) -> float:
        """获取平均置信度"""
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)
        
    def update_state(self, new_state: UserState):
        """更新用户状态"""
        if new_state != self.state:
            self.state_history.append({
                "from_state": self.state.value,
                "to_state": new_state.value,
                "timestamp": time.time()
            })
            self.state = new_state
            self.last_state_change = time.time()
            
    def to_dict(self) -> Dict:
        """转换为字典格式（用于持久化）"""
        # 安全地转换face_embeddings
        face_embeddings_list = []
        for emb in self.face_embeddings:
            if isinstance(emb, np.ndarray):
                face_embeddings_list.append(emb.tolist())
            elif isinstance(emb, list):
                face_embeddings_list.append(emb)
            else:
                # 如果是其他类型，尝试转换为numpy数组再转为列表
                try:
                    face_embeddings_list.append(np.array(emb).tolist())
                except:
                    # 如果转换失败，跳过这个特征
                    continue
                    
        return {
            "user_id": self.user_id,
            "name": self.name,
            "face_embeddings": face_embeddings_list,
            "last_seen": self.last_seen,
            "visit_count": self.visit_count,
            "confidence_history": list(self.confidence_history),
            "state": self.state.value,
            "last_state_change": self.last_state_change,
            "state_history": self.state_history
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """从字典创建用户档案"""
        return cls(
            user_id=data["user_id"],
            name=data["name"],
            face_embeddings=[np.array(emb) for emb in data["face_embeddings"]],
            last_seen=data["last_seen"],
            visit_count=data["visit_count"],
            confidence_history=deque(data["confidence_history"], maxlen=10),
            state=UserState(data["state"]),
            last_state_change=data["last_state_change"],
            state_history=data["state_history"]
        )

class UserManager:
    def __init__(self, config: Dict[str, Any]):
        """初始化用户管理器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 配置参数
        self.max_embeddings_per_user = config.get('max_embeddings_per_user', 5)  # 每个用户最多存储5个人脸特征
        self.face_threshold = config.get('face_threshold', 0.7)  # 人脸识别阈值
        
        # 用户数据存储
        self.users: Dict[str, UserProfile] = {}
        self.visitor_counter = 0
        self.last_recognized_user: Optional[UserProfile] = None
        
        # 初始化声纹识别器
        self.voice_recognizer = VoiceRecognizer(config)
        
        # 加载用户数据
        self._load_users()
        
    def _load_users(self):
        """加载持久化的用户数据"""
        try:
            data_dir = os.path.join('data', 'users')
            os.makedirs(data_dir, exist_ok=True)
            
            for filename in os.listdir(data_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(data_dir, filename), 'r') as f:
                        user_data = json.load(f)
                        user = UserProfile.from_dict(user_data)
                        self.users[user.user_id] = user
                        self.logger.info(f"加载用户数据: {user.user_id}")
                        
            # 更新访客计数器
            visitor_ids = [uid for uid in self.users.keys() if uid.startswith('visitor_')]
            if visitor_ids:
                max_visitor_num = max(int(uid.split('_')[1]) for uid in visitor_ids)
                self.visitor_counter = max_visitor_num
                
        except Exception as e:
            self.logger.error(f"加载用户数据失败: {e}")
            
    def _save_user(self, user: UserProfile):
        """保存用户数据"""
        try:
            data_dir = os.path.join('data', 'users')
            os.makedirs(data_dir, exist_ok=True)
            
            file_path = os.path.join(data_dir, f"{user.user_id}.json")
            with open(file_path, 'w') as f:
                json.dump(user.to_dict(), f)
                
            self.logger.debug(f"保存用户数据: {user.user_id}")
            
        except Exception as e:
            self.logger.error(f"保存用户数据失败: {e}")
            
    def _create_new_visitor(self) -> UserProfile:
        """创建新访客"""
        self.visitor_counter += 1
        visitor_id = f"visitor_{self.visitor_counter:03d}"
        self.logger.info(f"创建新用户: {visitor_id}")
        return self._create_new_user(None, visitor_id)
        
    def identify_user(self, perception_results: Dict[str, Any]) -> str:
        """识别用户"""
        current_time = time.time()
        
        # 检查是否检测到人脸和声音
        face_results = perception_results.get("face", {})
        voice_results = perception_results.get("voice", {})
        
        self.logger.debug(f"人脸检测结果: {face_results}")
        self.logger.debug(f"声纹检测结果: {voice_results}")
        
        # 如果同时有人脸和声音
        if face_results.get("detected") and voice_results.get("detected"):
            return self._identify_by_multimodal(face_results, voice_results)
            
        # 如果只有人脸
        if face_results.get("detected"):
            return self._identify_by_face(face_results)
            
        # 如果只有声音
        if voice_results.get("detected"):
            return self._identify_by_voice(voice_results)
            
        # 如果都没有检测到
        if self.last_recognized_user:
            self.last_recognized_user.update_state(UserState.INACTIVE)
            self._save_user(self.last_recognized_user)
            self.logger.info(f"未检测到任何特征，用户 {self.last_recognized_user.user_id} 状态更新为 INACTIVE")
            return self.last_recognized_user.user_id
            
        self.logger.info("未检测到任何特征，返回默认visitor")
        return "visitor"
        
    def _identify_by_voice(self, voice_results: Dict[str, Any]) -> str:
        """通过声纹识别用户"""
        voice_feature = voice_results.get("voice_embedding")
        if voice_feature is None:
            return "visitor"
            
        # 匹配声纹
        user_id, similarity = self.voice_recognizer.match_voice(voice_feature)
        
        if user_id and similarity >= self.voice_recognizer.voice_threshold:
            # 更新用户状态
            user = self.users.get(user_id)
            if user:
                user.update_state(UserState.ACTIVE)
                self._save_user(user)
                self.last_recognized_user = user
                
                # 更新声纹特征
                self.voice_recognizer.update_voice_profile(user_id, voice_feature)
                
                self.logger.info(f"通过声纹识别到用户 {user_id}, 置信度: {similarity:.2f}")
                return user_id
                
        # 如果没有匹配的用户，创建新访客
        new_visitor = self._create_new_visitor()
        self.voice_recognizer.update_voice_profile(new_visitor.user_id, voice_feature)
        self.last_recognized_user = new_visitor
        self.logger.info(f"创建新访客: {new_visitor.user_id}")
        return new_visitor.user_id
        
    def _identify_by_multimodal(self, face_results: Dict[str, Any], voice_results: Dict[str, Any]) -> str:
        """通过多模态识别用户
        
        Args:
            face_results: 人脸识别结果
            voice_results: 声纹识别结果
            
        Returns:
            识别的用户ID
        """
        # 获取人脸特征和声纹特征
        face_embedding = face_results.get("face_embedding")
        voice_feature = voice_results.get("voice_embedding")
        
        if face_embedding is None or voice_feature is None:
            return "visitor"
            
        # 人脸识别
        face_user_id = None
        face_similarity = 0.0
        for user in self.users.values():
            if not user.face_embeddings:
                continue
            similarities = []
            for stored_embedding in user.face_embeddings:
                similarity = compute_cosine_similarity(face_embedding, stored_embedding)
                similarities.append(similarity)
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity > face_similarity:
                    face_similarity = avg_similarity
                    face_user_id = user.user_id
                    
        # 声纹识别
        voice_user_id, voice_similarity = self.voice_recognizer.match_voice(voice_feature)
        
        # 多模态融合
        if face_user_id == voice_user_id and face_user_id is not None:
            # 两个模态都识别为同一用户
            user = self.users.get(face_user_id)
            if user:
                # 计算融合置信度
                fusion_confidence = self._calculate_fusion_confidence(
                    face_similarity,
                    voice_similarity,
                    face_results,
                    voice_results
                )
                
                if fusion_confidence > 0.7:  # 融合置信度阈值
                    user.update_state(UserState.ACTIVE)
                    self._save_user(user)
                    self.last_recognized_user = user
                    
                    # 更新特征
                    if face_similarity > 0.8:
                        self._add_face_embedding(user, face_embedding)
                    if voice_similarity > 0.8:
                        self.voice_recognizer.update_voice_profile(user.user_id, voice_feature)
                        
                    self.logger.info(
                        f"多模态识别到用户 {face_user_id}, "
                        f"人脸置信度: {face_similarity:.2f}, "
                        f"声纹置信度: {voice_similarity:.2f}, "
                        f"融合置信度: {fusion_confidence:.2f}"
                    )
                    return face_user_id
                    
        # 如果只有一个模态识别成功
        if face_similarity > self.face_threshold:
            return self._handle_face_match(face_user_id, face_embedding, face_similarity)
        elif voice_similarity > self.voice_recognizer.voice_threshold:
            return self._handle_voice_match(voice_user_id, voice_feature, voice_similarity)
            
        # 如果都没有匹配，创建新访客
        new_visitor = self._create_new_visitor()
        self._add_face_embedding(new_visitor, face_embedding)
        self.voice_recognizer.update_voice_profile(new_visitor.user_id, voice_feature)
        self.last_recognized_user = new_visitor
        self.logger.info(f"创建新访客: {new_visitor.user_id}")
        return new_visitor.user_id
        
    def _calculate_fusion_confidence(
        self,
        face_similarity: float,
        voice_similarity: float,
        face_results: Dict[str, Any],
        voice_results: Dict[str, Any]
    ) -> float:
        """计算多模态融合置信度
        
        Args:
            face_similarity: 人脸相似度
            voice_similarity: 声纹相似度
            face_results: 人脸识别结果
            voice_results: 声纹识别结果
            
        Returns:
            融合置信度
        """
        # 1. 计算模态权重
        face_weight = self._calculate_face_weight(face_results)
        voice_weight = self._calculate_voice_weight(voice_results)
        
        # 2. 计算加权融合置信度
        fusion_confidence = (
            face_weight * face_similarity +
            voice_weight * voice_similarity
        )
        
        return float(fusion_confidence)
        
    def _calculate_face_weight(self, face_results: Dict[str, Any]) -> float:
        """计算人脸模态权重
        
        Args:
            face_results: 人脸识别结果
            
        Returns:
            人脸模态权重
        """
        # 获取人脸姿态
        face_pose = face_results.get("pose", {})
        
        # 计算姿态权重
        pose_weight = self._calculate_pose_weight(face_pose)
        
        # 获取人脸质量分数
        face_quality = face_results.get("quality_score", 0.5)
        
        # 计算最终权重
        weight = 0.6 * pose_weight + 0.4 * face_quality
        
        return float(weight)
        
    def _calculate_voice_weight(self, voice_results: Dict[str, Any]) -> float:
        """计算声纹模态权重
        
        Args:
            voice_results: 声纹识别结果
            
        Returns:
            声纹模态权重
        """
        # 获取声纹质量分数
        voice_quality = voice_results.get("quality_score", 0.5)
        
        # 获取声纹置信度
        voice_confidence = voice_results.get("confidence", 0.5)
        
        # 计算最终权重
        weight = 0.5 * voice_quality + 0.5 * voice_confidence
        
        return float(weight)
        
    def _handle_face_match(self, user_id: str, face_embedding: np.ndarray, similarity: float) -> str:
        """处理人脸匹配结果"""
        user = self.users.get(user_id)
        if user:
            user.update_state(UserState.ACTIVE)
            self._save_user(user)
            self.last_recognized_user = user
            
            if similarity > 0.8:
                self._add_face_embedding(user, face_embedding)
                
            self.logger.info(f"通过人脸识别到用户 {user_id}, 置信度: {similarity:.2f}")
            return user_id
            
        return "visitor"
        
    def _handle_voice_match(self, user_id: str, voice_feature: np.ndarray, similarity: float) -> str:
        """处理声纹匹配结果"""
        user = self.users.get(user_id)
        if user:
            user.update_state(UserState.ACTIVE)
            self._save_user(user)
            self.last_recognized_user = user
            
            self.voice_recognizer.update_voice_profile(user_id, voice_feature)
            
            self.logger.info(f"通过声纹识别到用户 {user_id}, 置信度: {similarity:.2f}")
            return user_id
            
        return "visitor"
        
    def _calculate_pose_weight(self, pose: Dict[str, float]) -> float:
        """计算姿态权重，范围0-1"""
        if not pose:
            return 0.0
            
        # 获取姿态角度
        yaw = abs(pose.get("yaw", 0))
        pitch = abs(pose.get("pitch", 0))
        roll = abs(pose.get("roll", 0))
        
        # 计算姿态得分
        max_angle = 45.0  # 最大允许角度
        yaw_score = max(0, 1 - yaw / max_angle)
        pitch_score = max(0, 1 - pitch / max_angle)
        roll_score = max(0, 1 - roll / max_angle)
        
        # 综合得分，偏航角权重更高
        weight = (yaw_score * 0.5 + pitch_score * 0.3 + roll_score * 0.2)
        
        self.logger.debug(f"姿态得分 - 偏航: {yaw_score:.2f}, 俯仰: {pitch_score:.2f}, 翻滚: {roll_score:.2f}, 总权重: {weight:.2f}")
        
        return weight
        
    def _check_face_pose(self, pose: Dict[str, float]) -> bool:
        """检查人脸姿态是否合适"""
        if not pose:
            self.logger.debug("无姿态数据")
            return False
            
        # 检查偏航角（左右转动）
        yaw = abs(pose.get("yaw", 0))
        # 检查俯仰角（上下转动）
        pitch = abs(pose.get("pitch", 0))
        # 检查翻滚角（倾斜）
        roll = abs(pose.get("roll", 0))
        
        # 姿态角度都在阈值范围内
        is_valid = (yaw < self.pose_threshold and 
                   pitch < self.pose_threshold and 
                   roll < self.pose_threshold)
                   
        self.logger.debug(f"姿态检查 - 偏航角: {yaw:.1f}, 俯仰角: {pitch:.1f}, 翻滚角: {roll:.1f}, 是否有效: {is_valid}")
                   
        # 更新姿态历史
        if self.last_recognized_user:
            user_id = self.last_recognized_user.user_id
            if user_id not in self.pose_history:
                self.pose_history[user_id] = []
            self.pose_history[user_id].append((yaw, pitch, roll))
            if len(self.pose_history[user_id]) > self.max_pose_history:
                self.pose_history[user_id].pop(0)
                
        return is_valid
        
    def _is_good_pose_for_storage(self, pose: Dict[str, float]) -> bool:
        """检查姿态是否适合存储特征"""
        if not pose:
            return False
            
        # 检查偏航角（左右转动）
        yaw = abs(pose.get("yaw", 0))
        # 检查俯仰角（上下转动）
        pitch = abs(pose.get("pitch", 0))
        # 检查翻滚角（倾斜）
        roll = abs(pose.get("roll", 0))
        
        # 存储时使用更严格的姿态要求
        storage_threshold = 31.5  # 降低存储姿态要求
        
        # 姿态角度都在阈值范围内
        is_valid = (yaw < storage_threshold and 
                   pitch < storage_threshold and 
                   roll < storage_threshold)
                   
        self.logger.debug(f"存储姿态检查 - 偏航角: {yaw:.1f}, 俯仰角: {pitch:.1f}, 翻滚角: {roll:.1f}, 是否有效: {is_valid}")
                
        return is_valid
        
    def _add_face_embedding(self, user: UserProfile, embedding: np.ndarray):
        """添加人脸特征"""
        user.face_embeddings.append(embedding)
        # 如果特征数量超过限制，删除最旧的
        if len(user.face_embeddings) > self.max_embeddings_per_user:
            user.face_embeddings.pop(0)
            
    def _check_stability(self, user: UserProfile) -> bool:
        """检查用户识别的稳定性"""
        if len(user.confidence_history) < 3:  # 至少需要3次记录
            return True
            
        # 检查最近3次的置信度是否稳定
        recent_confidences = list(user.confidence_history)[-3:]
        avg_confidence = sum(recent_confidences) / len(recent_confidences)
        confidence_std = np.std(recent_confidences)
        
        return avg_confidence >= self.min_confidence and confidence_std < 0.15
        
    def _create_new_user(self, face_embedding: np.ndarray, user_id: str = None) -> UserProfile:
        """创建新用户"""
        if user_id is None:
            self.unknown_user_counter += 1
            user_id = f"user_{self.unknown_user_counter}"
        
        new_user = UserProfile(
            user_id=user_id,
            name=f"User {user_id}",
            face_embeddings=[face_embedding],
            last_seen=time.time(),
            visit_count=1,
            state=UserState.IDENTIFIED
        )
        
        self.users[user_id] = new_user
        self.logger.info(f"创建新用户: {user_id}")
        return new_user
        
    def register_face(self, user_id: str, face_embedding: np.ndarray, pose: Dict[str, float] = None):
        """注册人脸特征"""
        # 检查姿态
        if pose and not self._is_good_pose_for_storage(pose):
            return False
            
        if user_id not in self.users:
            self.users[user_id] = UserProfile(
                user_id=user_id,
                name=f"User {user_id}",
                face_embeddings=[face_embedding],
                last_seen=time.time(),
                visit_count=1
            )
        else:
            # 添加新的面部特征
            self._add_face_embedding(self.users[user_id], face_embedding)
            
        return True
        
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """更新用户画像"""
        try:
            # 如果用户不存在，自动创建
            if user_id not in self.users:
                if user_id == "visitor":
                    # 创建访客用户
                    self._create_visitor_user()
                else:
                    # 创建普通用户
                    face_info = profile_data.get('face_info', {})
                    face_embedding = face_info.get('face_embedding')
                    if face_embedding is not None:
                        self._create_new_user(face_embedding, user_id)
                    else:
                        # 创建没有人脸特征的用户
                        self.users[user_id] = UserProfile(
                            user_id=user_id,
                            name=f"User {user_id}",
                            face_embeddings=[],
                            last_seen=time.time(),
                            visit_count=1,
                            state=UserState.UNKNOWN
                        )
                        self.logger.info(f"创建新用户（无人脸特征）: {user_id}")
                
            user = self.users[user_id]
            
            # 更新基本信息
            if "name" in profile_data:
                user.name = profile_data["name"]
                
            # 更新访问次数
            if "visit_count" in profile_data:
                user.visit_count = profile_data["visit_count"]
            else:
                user.visit_count += 1
                
            # 更新最后见面时间
            user.last_seen = time.time()
            
            # 更新人脸特征（如果有）
            face_info = profile_data.get('face_info', {})
            if face_info.get('detected') and face_info.get('face_embedding') is not None:
                self._add_face_embedding(user, face_info['face_embedding'])
                
            # 更新声纹特征（如果有）
            voice_feature = profile_data.get('voice_feature')
            if voice_feature is not None:
                self.voice_recognizer.update_voice_profile(user_id, voice_feature)
            
            # 保存更新
            self._save_user(user)
            self.logger.debug(f"更新用户画像: {user_id}")
            
        except Exception as e:
            self.logger.error(f"更新用户画像失败: {e}")
            
    def _create_visitor_user(self):
        """创建访客用户"""
        if "visitor" not in self.users:
            self.users["visitor"] = UserProfile(
                user_id="visitor",
                name="访客",
                face_embeddings=[],
                last_seen=time.time(),
                visit_count=1,
                state=UserState.UNKNOWN
            )
            self.logger.info("创建访客用户")
            
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        获取用户画像信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户画像字典，如果用户不存在返回None
        """
        try:
            user = self.users.get(user_id)
            if not user:
                self.logger.warning(f"用户不存在: {user_id}")
                return None
                
            # 构建用户画像
            profile = {
                "user_id": user.user_id,
                "name": user.name,
                "visit_count": user.visit_count,
                "last_seen": datetime.fromtimestamp(user.last_seen).strftime("%Y-%m-%d %H:%M:%S"),
                "state": user.state.value,
                "average_confidence": user.get_average_confidence(),
                "is_frequent_visitor": user.visit_count >= 5,
                "tags": []
            }
            
            # 根据访问频率添加标签
            if user.visit_count >= 10:
                profile["tags"].append("常客")
            elif user.visit_count >= 5:
                profile["tags"].append("熟悉用户")
            elif user.visit_count >= 2:
                profile["tags"].append("回头客")
            else:
                profile["tags"].append("新用户")
                
            # 根据状态添加标签
            if user.state == UserState.ACTIVE:
                profile["tags"].append("活跃")
            elif user.state == UserState.INACTIVE:
                profile["tags"].append("离线")
                
            # 根据识别置信度添加标签
            avg_confidence = user.get_average_confidence()
            if avg_confidence >= 0.9:
                profile["tags"].append("高识别度")
            elif avg_confidence >= 0.7:
                profile["tags"].append("中识别度")
            else:
                profile["tags"].append("待确认")
                
            return profile
            
        except Exception as e:
            self.logger.error(f"获取用户画像失败: {e}")
            return None
            
    def get_all_users(self) -> List[Dict[str, Any]]:
        """
        获取所有用户的基本信息
        
        Returns:
            用户信息列表
        """
        try:
            users_info = []
            for user_id, user in self.users.items():
                users_info.append({
                    "user_id": user.user_id,
                    "name": user.name,
                    "visit_count": user.visit_count,
                    "state": user.state.value,
                    "last_seen": datetime.fromtimestamp(user.last_seen).strftime("%Y-%m-%d %H:%M:%S")
                })
            return users_info
        except Exception as e:
            self.logger.error(f"获取用户列表失败: {e}")
            return [] 