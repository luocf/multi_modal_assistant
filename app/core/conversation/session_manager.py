import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from loguru import logger

@dataclass
class Message:
    """会话消息"""
    role: str  # 'user' 或 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Topic:
    """话题信息"""
    name: str
    start_time: float
    last_active: float
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Session:
    """会话信息"""
    session_id: str
    user_id: str
    start_time: float
    last_active: float
    topics: List[Topic] = field(default_factory=list)
    current_topic: Optional[Topic] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class SessionManager:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化会话管理器
        Args:
            config: 配置字典，包含：
                - session_timeout: 会话超时时间（秒）
                - topic_timeout: 话题超时时间（秒）
                - max_history_messages: 最大历史消息数
                - storage_dir: 会话存储目录
        """
        self.config = config
        self.session_timeout = config.get("session_timeout", 3600)  # 1小时
        self.topic_timeout = config.get("topic_timeout", 300)  # 5分钟
        self.max_history_messages = config.get("max_history_messages", 50)
        self.storage_dir = config.get("storage_dir", "data/sessions")
        
        # 会话存储
        self.active_sessions: Dict[str, Session] = {}
        
        # 创建存储目录
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # 加载持久化的会话
        self._load_sessions()
        
    def create_session(self, user_id: str) -> Session:
        """创建新会话"""
        session_id = f"{user_id}_{int(time.time())}"
        current_time = time.time()
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            start_time=current_time,
            last_active=current_time
        )
        
        self.active_sessions[session_id] = session
        logger.info(f"创建新会话: {session_id}")
        return session
        
    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        session = self.active_sessions.get(session_id)
        if session and self._is_session_expired(session):
            self.end_session(session_id)
            return None
        return session
        
    def get_user_session(self, user_id: str) -> Optional[Session]:
        """获取用户的活跃会话"""
        for session in self.active_sessions.values():
            if session.user_id == user_id and not self._is_session_expired(session):
                return session
        return None
        
    def end_session(self, session_id: str):
        """结束会话"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            # 保存会话
            self._save_session(session)
            # 从活跃会话中移除
            del self.active_sessions[session_id]
            logger.info(f"结束会话: {session_id}")
            
    def add_message(self, session_id: str, role: str, content: str, 
                   metadata: Dict[str, Any] = None) -> Optional[Message]:
        """添加消息到会话"""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"会话不存在或已过期: {session_id}")
            return None
            
        # 创建消息
        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # 如果没有当前话题或当前话题已过期，创建新话题
        if not session.current_topic or self._is_topic_expired(session.current_topic):
            topic = Topic(
                name=f"Topic_{len(session.topics) + 1}",
                start_time=time.time(),
                last_active=time.time()
            )
            session.topics.append(topic)
            session.current_topic = topic
            
        # 添加消息到当前话题
        session.current_topic.messages.append(message)
        session.current_topic.last_active = time.time()
        session.last_active = time.time()
        
        # 限制历史消息数量
        if len(session.current_topic.messages) > self.max_history_messages:
            session.current_topic.messages = session.current_topic.messages[-self.max_history_messages:]
            
        return message
        
    def get_context(self, session_id: str, max_messages: int = None) -> List[Message]:
        """获取会话上下文"""
        session = self.get_session(session_id)
        if not session or not session.current_topic:
            return []
            
        messages = session.current_topic.messages
        if max_messages:
            messages = messages[-max_messages:]
            
        return messages
        
    def get_topic_summary(self, topic: Topic) -> str:
        """生成话题摘要"""
        if not topic.messages:
            return ""
            
        # 简单实现：使用最近几条消息作为摘要
        recent_messages = topic.messages[-3:]
        summary = []
        for msg in recent_messages:
            summary.append(f"{msg.role}: {msg.content[:50]}...")
        return "\n".join(summary)
        
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """获取会话摘要"""
        session = self.get_session(session_id)
        if not session:
            return {}
            
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "duration": time.time() - session.start_time,
            "topics": [
                {
                    "name": topic.name,
                    "summary": self.get_topic_summary(topic),
                    "message_count": len(topic.messages),
                    "duration": topic.last_active - topic.start_time
                }
                for topic in session.topics
            ],
            "total_messages": sum(len(topic.messages) for topic in session.topics)
        }
        
    def _is_session_expired(self, session: Session) -> bool:
        """检查会话是否过期"""
        return time.time() - session.last_active > self.session_timeout
        
    def _is_topic_expired(self, topic: Topic) -> bool:
        """检查话题是否过期"""
        return time.time() - topic.last_active > self.topic_timeout
        
    def _save_session(self, session: Session):
        """保存会话到文件"""
        try:
            # 构建文件路径
            file_path = os.path.join(
                self.storage_dir,
                f"{session.user_id}",
                f"{session.session_id}.json"
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 转换为可序列化的字典
            session_data = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "start_time": session.start_time,
                "last_active": session.last_active,
                "topics": [
                    {
                        "name": topic.name,
                        "start_time": topic.start_time,
                        "last_active": topic.last_active,
                        "messages": [
                            {
                                "role": msg.role,
                                "content": msg.content,
                                "timestamp": msg.timestamp,
                                "metadata": msg.metadata
                            }
                            for msg in topic.messages
                        ],
                        "metadata": topic.metadata
                    }
                    for topic in session.topics
                ],
                "metadata": session.metadata
            }
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"会话保存成功: {file_path}")
            
        except Exception as e:
            logger.error(f"会话保存失败: {e}")
            
    def _load_sessions(self):
        """加载持久化的会话"""
        try:
            # 遍历用户目录
            for user_dir in os.listdir(self.storage_dir):
                user_path = os.path.join(self.storage_dir, user_dir)
                if not os.path.isdir(user_path):
                    continue
                    
                # 遍历会话文件
                for session_file in os.listdir(user_path):
                    if not session_file.endswith(".json"):
                        continue
                        
                    file_path = os.path.join(user_path, session_file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            session_data = json.load(f)
                            
                        # 重建会话对象
                        session = Session(
                            session_id=session_data["session_id"],
                            user_id=session_data["user_id"],
                            start_time=session_data["start_time"],
                            last_active=session_data["last_active"],
                            metadata=session_data["metadata"]
                        )
                        
                        # 重建话题和消息
                        for topic_data in session_data["topics"]:
                            topic = Topic(
                                name=topic_data["name"],
                                start_time=topic_data["start_time"],
                                last_active=topic_data["last_active"],
                                metadata=topic_data["metadata"]
                            )
                            
                            for msg_data in topic_data["messages"]:
                                message = Message(
                                    role=msg_data["role"],
                                    content=msg_data["content"],
                                    timestamp=msg_data["timestamp"],
                                    metadata=msg_data["metadata"]
                                )
                                topic.messages.append(message)
                                
                            session.topics.append(topic)
                            
                        # 如果会话未过期，添加到活跃会话
                        if not self._is_session_expired(session):
                            self.active_sessions[session.session_id] = session
                            
                    except Exception as e:
                        logger.error(f"加载会话文件失败 {file_path}: {e}")
                        
            logger.info(f"加载了 {len(self.active_sessions)} 个活跃会话")
            
        except Exception as e:
            logger.error(f"加载会话失败: {e}")
            
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        expired_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if self._is_session_expired(session)
        ]
        
        for session_id in expired_sessions:
            self.end_session(session_id)
            
        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期会话") 