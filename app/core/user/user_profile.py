from typing import List, Optional, Dict, Any
import time
import numpy as np

class UserProfile:
    def __init__(self, user_id: str, name: Optional[str] = None):
        """初始化用户配置文件
        
        Args:
            user_id: 用户ID
            name: 用户名称（可选）
        """
        self.user_id = user_id
        self.name = name or user_id
        self.face_embeddings: List[np.ndarray] = []  # 人脸特征列表
        self.voice_embeddings: List[np.ndarray] = []  # 声纹特征列表
        self.state = UserState.INACTIVE
        self.last_active_time = 0
        self.created_time = time.time()
        self.metadata: Dict[str, Any] = {}  # 其他元数据
        
    def update_state(self, state: UserState):
        """更新用户状态
        
        Args:
            state: 新的用户状态
        """
        self.state = state
        if state == UserState.ACTIVE:
            self.last_active_time = time.time()
            
    def add_face_embedding(self, embedding: np.ndarray):
        """添加人脸特征
        
        Args:
            embedding: 人脸特征向量
        """
        self.face_embeddings.append(embedding)
        
    def add_voice_embedding(self, embedding: np.ndarray):
        """添加声纹特征
        
        Args:
            embedding: 声纹特征向量
        """
        self.voice_embeddings.append(embedding)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            字典格式的用户配置
        """
        return {
            "user_id": self.user_id,
            "name": self.name,
            "face_embeddings": [emb.tolist() for emb in self.face_embeddings],
            "voice_embeddings": [emb.tolist() for emb in self.voice_embeddings],
            "state": self.state.value,
            "last_active_time": self.last_active_time,
            "created_time": self.created_time,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """从字典创建用户配置
        
        Args:
            data: 字典格式的用户配置
            
        Returns:
            用户配置对象
        """
        profile = cls(data["user_id"], data["name"])
        profile.face_embeddings = [np.array(emb) for emb in data["face_embeddings"]]
        profile.voice_embeddings = [np.array(emb) for emb in data["voice_embeddings"]]
        profile.state = UserState(data["state"])
        profile.last_active_time = data["last_active_time"]
        profile.created_time = data["created_time"]
        profile.metadata = data["metadata"]
        return profile 