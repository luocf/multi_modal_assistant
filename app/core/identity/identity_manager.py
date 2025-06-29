from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import time
from datetime import datetime
import json
import os
from loguru import logger
from dataclasses import dataclass, field
import uuid
from app.core.utils import compute_cosine_similarity, encrypt_data, decrypt_data

@dataclass
class BiometricProfile:
    """生物特征档案"""
    face_embeddings: List[np.ndarray] = field(default_factory=list)  # 人脸特征向量列表
    voice_embeddings: List[np.ndarray] = field(default_factory=list)  # 声纹特征向量列表
    face_timestamps: List[float] = field(default_factory=list)  # 人脸特征更新时间
    voice_timestamps: List[float] = field(default_factory=list)  # 声纹特征更新时间
    last_face_seen: Optional[float] = None  # 最后看到的时间
    last_voice_heard: Optional[float] = None  # 最后听到的时间
    
    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            "face_embeddings": [emb.tolist() for emb in self.face_embeddings],
            "voice_embeddings": [emb.tolist() for emb in self.voice_embeddings],
            "face_timestamps": self.face_timestamps,
            "voice_timestamps": self.voice_timestamps,
            "last_face_seen": self.last_face_seen,
            "last_voice_heard": self.last_voice_heard
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'BiometricProfile':
        """从字典创建实例"""
        return cls(
            face_embeddings=[np.array(emb) for emb in data["face_embeddings"]],
            voice_embeddings=[np.array(emb) for emb in data["voice_embeddings"]],
            face_timestamps=data["face_timestamps"],
            voice_timestamps=data["voice_timestamps"],
            last_face_seen=data["last_face_seen"],
            last_voice_heard=data["last_voice_heard"]
        )

@dataclass
class UserProfile:
    """用户档案"""
    user_id: str
    name: Optional[str] = None
    biometric: BiometricProfile = field(default_factory=BiometricProfile)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "biometric": self.biometric.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_active": self.last_active
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        """从字典创建实例"""
        return cls(
            user_id=data["user_id"],
            name=data["name"],
            biometric=BiometricProfile.from_dict(data["biometric"]),
            metadata=data["metadata"],
            created_at=data["created_at"],
            last_active=data["last_active"]
        )

class IdentityManager:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化身份管理器
        Args:
            config: 配置字典，包含：
                - face_similarity_threshold: 人脸相似度阈值
                - voice_similarity_threshold: 声纹相似度阈值
                - feature_max_age: 特征向量最大年龄（秒）
                - max_features_per_user: 每个用户最大特征数量
                - storage_dir: 存储目录
        """
        self.config = config
        self.face_threshold = config.get("face_similarity_threshold", 0.6)
        self.voice_threshold = config.get("voice_similarity_threshold", 0.75)
        self.feature_max_age = config.get("feature_max_age", 30 * 24 * 3600)  # 30天
        self.max_features = config.get("max_features_per_user", 10)
        self.storage_dir = config.get("storage_dir", "data/identities")
        
        # 用户档案存储
        self.profiles: Dict[str, UserProfile] = {}
        
        # 创建存储目录
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # 加载现有档案
        self._load_profiles()
        
    def identify_face(self, face_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        识别人脸对应的用户ID
        Returns:
            (user_id, confidence) 或 (None, 0.0)
        """
        best_match = None
        best_score = 0.0
        
        for user_id, profile in self.profiles.items():
            if not profile.biometric.face_embeddings:
                continue
                
            # 计算与所有特征向量的相似度
            similarities = [
                compute_cosine_similarity(face_embedding, stored_emb)
                for stored_emb in profile.biometric.face_embeddings
            ]
            
            # 使用最高相似度
            max_similarity = max(similarities)
            if max_similarity > best_score:
                best_score = max_similarity
                best_match = user_id
                
        if best_score >= self.face_threshold:
            return best_match, best_score
        return None, 0.0
        
    def identify_voice(self, voice_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        识别声纹对应的用户ID
        Returns:
            (user_id, confidence) 或 (None, 0.0)
        """
        best_match = None
        best_score = 0.0
        
        for user_id, profile in self.profiles.items():
            if not profile.biometric.voice_embeddings:
                continue
                
            # 计算与所有特征向量的相似度
            similarities = [
                compute_cosine_similarity(voice_embedding, stored_emb)
                for stored_emb in profile.biometric.voice_embeddings
            ]
            
            # 使用最高相似度
            max_similarity = max(similarities)
            if max_similarity > best_score:
                best_score = max_similarity
                best_match = user_id
                
        if best_score >= self.voice_threshold:
            return best_match, best_score
        return None, 0.0
        
    def fuse_identities(
        self,
        face_embedding: Optional[np.ndarray],
        voice_embedding: Optional[np.ndarray],
        emotion: Optional[dict] = None,
        gesture: Optional[dict] = None,
        temporal_threshold: float = 2.0  # 时间阈值（秒）
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """
        融合人脸、声纹、表情、手势等多模态识别结果
        Returns:
            (user_id, confidences) 或 (None, {})
        """
        current_time = time.time()
        face_id, face_conf = (None, 0.0) if face_embedding is None else self.identify_face(face_embedding)
        voice_id, voice_conf = (None, 0.0) if voice_embedding is None else self.identify_voice(voice_embedding)
        # 表情和手势可选，暂以置信度最高的主表情/手势为特征
        emotion_type, emotion_conf = None, 0.0
        if emotion and emotion.get('emotion') and emotion.get('confidence', 0.0) > 0.5:
            emotion_type = emotion['emotion']
            emotion_conf = emotion['confidence']
        gesture_type, gesture_conf = None, 0.0
        if gesture and gesture.get('gesture') and gesture.get('confidence', 0.0) > 0.5:
            gesture_type = gesture['gesture']
            gesture_conf = gesture['confidence']
        # 多模态融合策略：优先人脸+声纹一致，否则置信度加权
        if face_id and voice_id and face_id == voice_id:
            profile = self.profiles[face_id]
            profile.biometric.last_face_seen = current_time
            profile.biometric.last_voice_heard = current_time
            return face_id, {"face": face_conf, "voice": voice_conf, "emotion": emotion_conf, "gesture": gesture_conf}
        # 置信度加权融合
        candidates = []
        if face_id:
            candidates.append((face_id, face_conf * 0.6))
        if voice_id:
            candidates.append((voice_id, voice_conf * 0.4))
        # 可扩展表情/手势加权
        # if emotion_type:
        #     candidates.append((face_id, emotion_conf * 0.1))
        # if gesture_type:
        #     candidates.append((face_id, gesture_conf * 0.1))
        if not candidates:
            return None, {}
        # 汇总加权分数
        score_map = {}
        for uid, score in candidates:
            score_map[uid] = score_map.get(uid, 0) + score
        best_id = max(score_map.items(), key=lambda x: x[1])[0]
        return best_id, {"face": face_conf, "voice": voice_conf, "emotion": emotion_conf, "gesture": gesture_conf}
        
    def update_profile(
        self,
        user_id: str,
        face_embedding: Optional[np.ndarray] = None,
        voice_embedding: Optional[np.ndarray] = None,
        emotion: Optional[dict] = None,
        gesture: Optional[dict] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """动态补全多模态特征"""
        if user_id not in self.profiles:
            logger.warning(f"用户 {user_id} 不存在，无法补全画像")
            return False
        profile = self.profiles[user_id]
        now = time.time()
        if face_embedding is not None:
            profile.biometric.face_embeddings.append(face_embedding)
            profile.biometric.face_timestamps.append(now)
            if len(profile.biometric.face_embeddings) > self.max_features:
                profile.biometric.face_embeddings = profile.biometric.face_embeddings[-self.max_features:]
                profile.biometric.face_timestamps = profile.biometric.face_timestamps[-self.max_features:]
        if voice_embedding is not None:
            profile.biometric.voice_embeddings.append(voice_embedding)
            profile.biometric.voice_timestamps.append(now)
            if len(profile.biometric.voice_embeddings) > self.max_features:
                profile.biometric.voice_embeddings = profile.biometric.voice_embeddings[-self.max_features:]
                profile.biometric.voice_timestamps = profile.biometric.voice_timestamps[-self.max_features:]
        if emotion is not None:
            profile.metadata['last_emotion'] = emotion
        if gesture is not None:
            profile.metadata['last_gesture'] = gesture
        if metadata:
            profile.metadata.update(metadata)
        profile.last_active = now
        logger.info(f"用户 {user_id} 画像已动态补全")
        return True
        
    def create_profile(
        self,
        name: Optional[str] = None,
        face_embedding: Optional[np.ndarray] = None,
        voice_embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """创建新的用户档案"""
        try:
            user_id = str(uuid.uuid4())
            profile = UserProfile(user_id=user_id, name=name, metadata=metadata or {})
            if face_embedding is not None:
                profile.biometric.face_embeddings.append(np.array(face_embedding, dtype=np.float64))
                profile.biometric.face_timestamps.append(time.time())
            if voice_embedding is not None:
                profile.biometric.voice_embeddings.append(np.array(voice_embedding, dtype=np.float64))
                profile.biometric.voice_timestamps.append(time.time())
            self.profiles[user_id] = profile
            logger.info(f"创建新用户档案: {user_id}")
            return user_id
            
        except Exception as e:
            logger.error(f"创建用户档案失败: {e}")
            return None
            
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户档案"""
        return self.profiles.get(user_id)
        
    def _save_profile(self, profile: UserProfile):
        """保存用户档案到文件"""
        try:
            file_path = os.path.join(self.storage_dir, f"{profile.user_id}.json")
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存用户档案失败: {e}")
            
    def _load_profiles(self):
        """加载所有用户档案"""
        try:
            for filename in os.listdir(self.storage_dir):
                if not filename.endswith(".json"):
                    continue
                    
                file_path = os.path.join(self.storage_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        profile = UserProfile.from_dict(data)
                        self.profiles[profile.user_id] = profile
                except Exception as e:
                    logger.error(f"加载用户档案失败 {file_path}: {e}")
                    
            logger.info(f"加载了 {len(self.profiles)} 个用户档案")
            
        except Exception as e:
            logger.error(f"加载用户档案失败: {e}")
            
    def cleanup_old_features(self):
        """清理过期的特征向量"""
        current_time = time.time()
        
        for profile in self.profiles.values():
            # 清理过期的人脸特征
            valid_face_indices = [
                i for i, ts in enumerate(profile.biometric.face_timestamps)
                if current_time - ts <= self.feature_max_age
            ]
            
            profile.biometric.face_embeddings = [
                profile.biometric.face_embeddings[i] for i in valid_face_indices
            ]
            profile.biometric.face_timestamps = [
                profile.biometric.face_timestamps[i] for i in valid_face_indices
            ]
            
            # 清理过期的声纹特征
            valid_voice_indices = [
                i for i, ts in enumerate(profile.biometric.voice_timestamps)
                if current_time - ts <= self.feature_max_age
            ]
            
            profile.biometric.voice_embeddings = [
                profile.biometric.voice_embeddings[i] for i in valid_voice_indices
            ]
            profile.biometric.voice_timestamps = [
                profile.biometric.voice_timestamps[i] for i in valid_voice_indices
            ]
            
    def save_profiles(self, encrypt=True, key=None):
        """加密保存所有用户画像"""
        data = json.dumps({uid: p.to_dict() for uid, p in self.profiles.items()}).encode()
        if encrypt:
            data = encrypt_data(data, key)
        with open(os.path.join(self.storage_dir, 'profiles.enc' if encrypt else 'profiles.json'), 'wb' if encrypt else 'w') as f:
            if encrypt:
                f.write(data)
            else:
                f.write(data.decode())
        logger.info(f"用户画像已{'加密' if encrypt else ''}保存")
    def load_profiles(self, decrypt=True, key=None):
        """加载加密用户画像"""
        path = os.path.join(self.storage_dir, 'profiles.enc' if decrypt else 'profiles.json')
        if not os.path.exists(path):
            logger.warning("画像文件不存在")
            return
        with open(path, 'rb' if decrypt else 'r') as f:
            data = f.read() if decrypt else f.read().encode()
        if decrypt:
            data = decrypt_data(data, key)
        profiles_dict = json.loads(data.decode())
        self.profiles = {uid: UserProfile.from_dict(p) for uid, p in profiles_dict.items()}
        logger.info(f"用户画像已{'解密' if decrypt else ''}加载")
    def delete_user_profile(self, user_id: str):
        """彻底删除指定用户画像"""
        if user_id in self.profiles:
            del self.profiles[user_id]
            logger.info(f"已彻底删除用户画像: {user_id}") 