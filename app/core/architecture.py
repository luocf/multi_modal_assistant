from typing import Dict, Any, Optional, List
import time
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

from datetime import datetime
from loguru import logger

from app.core.perception.face_recognizer import FaceRecognizer
from app.core.perception.voice_recognizer import VoiceRecognizer
from app.core.dialogue.dialogue_manager import DialogueManager
from app.core.user.user_manager import UserManager
from app.models.model_factory import ModelFactory
from app.config.user_config import USER_CONFIG
from app.config.voice_config import VOICE_CONFIG
from app.config.model_config import ModelType
from app.config.audio_config import AUDIO_CONFIG
from app.core.perception.emotion_recognizer import EmotionRecognizer
from app.core.perception.gesture_recognizer import GestureRecognizer
from app.core.utils import encrypt_data, decrypt_data

# 系统状态枚举
class SystemState(Enum):
    """系统状态枚举"""
    IDLE = "idle"           # 空闲状态
    LISTENING = "listening" # 正在听
    THINKING = "thinking"   # 正在思考
    SPEAKING = "speaking"   # 正在说话
    ERROR = "error"         # 错误状态

# 用户状态枚举
class UserState(Enum):
    UNKNOWN = "unknown"
    IDENTIFIED = "identified"
    ACTIVE = "active"
    INACTIVE = "inactive"

# 系统配置类
class SystemConfig:
    """系统配置"""
    def __init__(self):
        self.model_config = ModelType.LIGHT.value  # 使用枚举值
        self.audio_config = {
            "format": "int16",
            "channels": 1,
            "rate": 16000,
            "chunk": 1024
        }
        self.logger = logger
        
        self.recognition_config = {
            "face_confidence_threshold": 0.6,
        }
        
        self.performance_config = {
            "skip_frames": 2,
            "batch_size": 1
        }

# 系统管理器
class SystemManager:
    def __init__(self):
        self.state = SystemState.INITIALIZING
        self.config = SystemConfig()
        self.logger = logging.getLogger(__name__)
        
    def initialize(self):
        """初始化系统"""
        try:
            self._load_models()
            self._initialize_modules()
            self.state = SystemState.READY
        except Exception as e:
            self.logger.error(f"系统初始化失败: {str(e)}")
            self.state = SystemState.ERROR
            raise
            
    def _load_models(self):
        """加载模型"""
        pass
        
    def _initialize_modules(self):
        """初始化各个模块"""
        pass


# 对话引擎
class DialogueEngine:
    def __init__(self):
        """初始化对话引擎"""
        self.model = ModelFactory.create_model("light")  # 使用light模型
        self.last_response = "你好！我是你的AI助手。"
        self.is_processing = False
        self.response_cache = {}  # 缓存用户输入对应的响应
        self.cache_size = 100  # 缓存大小
        self.max_cache_age = 3600  # 缓存有效期（秒）
        self.last_context = None  # 记录上一次的上下文
        self.context_similarity_threshold = 0.8  # 上下文相似度阈值
        
    def process_input(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """处理用户输入"""
        if self.is_processing:
            return self.last_response
            
        # 检查上下文相似度
        if context and self.last_context:
            similarity = self._calculate_context_similarity(context, self.last_context)
            if similarity > self.context_similarity_threshold:
                return self.last_response
                
        # 检查缓存
        cache_key = f"{text}_{str(context)}"
        if cache_key in self.response_cache:
            cached_response, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < self.max_cache_age:
                return cached_response
                
        try:
            self.is_processing = True
            # 构建提示词
            prompt = self._build_prompt(text, context)
            
            # 生成回复
            response = self.model.generate(prompt)
            self.last_response = response
            self.last_context = context
            
            # 更新缓存
            self._update_cache(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"处理用户输入失败: {e}")
            return "抱歉，我现在无法回应，请稍后再试。"
        finally:
            self.is_processing = False
            
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """计算两个上下文的相似度"""
        if not context1 or not context2:
            return 0.0
            
        # 比较人脸检测结果
        face1 = context1.get("face", {})
        face2 = context2.get("face", {})
        
        if face1.get("detected") != face2.get("detected"):
            return 0.0
            
        if face1.get("detected"):
            # 如果都检测到人脸，比较置信度
            conf1 = face1.get("confidence", 0.0)
            conf2 = face2.get("confidence", 0.0)
            return 1.0 - abs(conf1 - conf2)
            
        return 1.0  # 如果都没有检测到人脸，认为上下文相似
        
    def _update_cache(self, key: str, response: str):
        """更新响应缓存"""
        # 如果缓存已满，删除最旧的条目
        if len(self.response_cache) >= self.cache_size:
            oldest_key = min(self.response_cache.keys(), 
                           key=lambda k: self.response_cache[k][1])
            del self.response_cache[oldest_key]
            
        self.response_cache[key] = (response, time.time())
        
    def _build_prompt(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """构建prompt"""
        prompt = f"User: {text}\n"
        if context and context.get("face", {}).get("detected"):
            prompt += "系统检测到人脸，请保持友好和专业的对话风格。\n"
        prompt += "Assistant: "
        return prompt

# 主应用类
class MultiModalAssistant:
    """多模态助手"""
    def __init__(self):
        """初始化多模态助手"""
        self.system_state = SystemState.IDLE
        self.last_response_time = datetime.now()
        self.is_running = False
        
        # 初始化各个组件
        self.dialogue_manager = DialogueManager()
        self.voice_recognizer = VoiceRecognizer(VOICE_CONFIG)
        
        # 为 FaceRecognizer 创建配置
        face_config = {
            "face_confidence_threshold": 0.6,
            "model_path": None,  # 使用默认模型
            "device": "cpu"
        }
        self.face_recognizer = FaceRecognizer(face_config)
        self.user_manager = UserManager(USER_CONFIG)
        self.emotion_recognizer = EmotionRecognizer({"model_type": "fer"})
        self.gesture_recognizer = GestureRecognizer({})
        self.identity_manager = None  # 可选：如有 identity_manager 模块
        
        logger.info("多模态助手初始化完成")
        
    def start(self):
        """启动系统"""
        if self.is_running:
            logger.warning("系统已经在运行")
            return
            
        self.is_running = True
        self.system_state = SystemState.IDLE
        logger.info("系统已启动")
        
    def stop(self):
        """停止系统"""
        if not self.is_running:
            logger.warning("系统已经停止")
            return
            
        self.is_running = False
        self.system_state = SystemState.IDLE
        logger.info("系统已停止")
        
    def process_input(self, text: str = None, audio: bytes = None, image: bytes = None) -> Optional[str]:
        """处理多模态输入，集成表情、手势、声纹、画像、加密等"""
        if not self.is_running:
            logger.error("系统未启动")
            return None
        try:
            multimodal_features = {}
            # 文本输入
            if text:
                self.system_state = SystemState.THINKING
                # 对于文本输入，使用最后识别的用户或访客
                current_user_id = self.user_manager.last_recognized_user.user_id if self.user_manager.last_recognized_user else 'visitor'
                user_profile = None
                if current_user_id != 'visitor' and self.user_manager:
                    user_profile = self.user_manager.get_user_profile(current_user_id)
                    
                response = self.dialogue_manager.generate_response(
                    prompt=text,
                    user_id=current_user_id,
                    user_profile=user_profile
                )
                self.system_state = SystemState.SPEAKING
                self.last_response_time = datetime.now()
                return response
            # 音频输入
            if audio:
                self.system_state = SystemState.THINKING
                audio_text = self.voice_recognizer.recognize(audio)
                voice_feature = self.voice_recognizer.extract_voice_feature(np.frombuffer(audio, dtype=np.int16).astype(np.float32))
                multimodal_features['voice'] = voice_feature
                user_id, voice_conf = self.voice_recognizer.match_voice(voice_feature)
                multimodal_features['voice_user'] = user_id
                
                # 打印声纹识别结果
                if user_id and user_id != 'unknown':
                    logger.info(f"🎤 声纹识别: {user_id} (置信度={voice_conf:.2f})")
                else:
                    logger.info("🎤 声纹识别: 未识别到已知用户")
                
                if audio_text:
                    logger.info(f"📝 语音转文字: {audio_text}")
                    
                    # 获取用户画像信息
                    user_profile = None
                    if user_id and user_id != 'unknown' and self.user_manager:
                        user_profile = self.user_manager.get_user_profile(user_id)
                        if user_profile:
                            logger.info(f"👤 用户画像: {user_profile['name']} (访问{user_profile['visit_count']}次, 标签: {', '.join(user_profile['tags'])})")
                    
                    # 生成个性化回复
                    response = self.dialogue_manager.generate_response(
                        audio_text=audio_text,
                        user_id=user_id,
                        user_profile=user_profile
                    )
                    self.system_state = SystemState.SPEAKING
                    self.last_response_time = datetime.now()
                    return response
                else:
                    logger.debug("未识别到语音内容")
            # 图像输入
            if image:
                self.system_state = SystemState.THINKING
                try:
                    nparr = np.frombuffer(image, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        # 人脸检测
                        face_info = self.face_recognizer.detect(frame)
                        multimodal_features['face'] = face_info
                        if face_info.get('detected'):
                            logger.info(f"👤 检测到人脸: 置信度={face_info.get('confidence', 0):.2f}")
                        
                        # 表情识别
                        emotion_info = self.emotion_recognizer.detect_emotion(frame)
                        multimodal_features['emotion'] = emotion_info
                        if emotion_info.get('detected'):
                            emotion = emotion_info.get('emotion', 'unknown')
                            confidence = emotion_info.get('confidence', 0)
                            logger.info(f"😊 检测到表情: {emotion} (置信度={confidence:.2f})")
                        
                        # 手势识别
                        gesture_info = self.gesture_recognizer.detect_gesture(frame)
                        multimodal_features['gesture'] = gesture_info
                        if gesture_info.get('detected'):
                            gesture = gesture_info.get('gesture', 'unknown')
                            confidence = gesture_info.get('confidence', 0)
                            logger.info(f"👋 检测到手势: {gesture} (置信度={confidence:.2f})")
                        # 多模态身份融合（如有 identity_manager）
                        if self.identity_manager:
                            user_id, confidences = self.identity_manager.fuse_identities(
                                face_embedding=face_info.get('face_embedding'),
                                voice_embedding=multimodal_features.get('voice'),
                                emotion=emotion_info,
                                gesture=gesture_info
                            )
                            multimodal_features['user_id'] = user_id
                        # 画像动态补全
                        if self.user_manager:
                            profile_data = {
                                'face_info': face_info,
                                'voice_feature': multimodal_features.get('voice'),
                                'emotion_info': emotion_info,
                                'gesture_info': gesture_info
                            }
                            user_id = multimodal_features.get('user_id', 'visitor')
                            self.user_manager.update_user_profile(user_id, profile_data)
                        # 只记录视觉信息，不调用大模型
                        logger.debug("已记录视觉信息，等待用户语音输入")
                except Exception as e:
                    logger.error(f"图像处理失败: {e}")
            self.system_state = SystemState.IDLE
            return None
        except Exception as e:
            logger.error(f"处理输入失败: {e}")
            self.system_state = SystemState.ERROR
            return None
            
    def get_system_state(self) -> SystemState:
        """获取系统状态"""
        return self.system_state
        
    def get_last_response_time(self) -> datetime:
        """获取最后一次响应时间"""
        return self.last_response_time 

    def process_input_stream(self, text: str = None, audio: bytes = None, image: bytes = None):
        """处理多模态输入，流式输出"""
        if not self.is_running:
            logger.error("系统未启动")
            yield None
            return
        try:
            if text:
                self.system_state = SystemState.THINKING
                # 对于文本输入，使用最后识别的用户或访客
                current_user_id = self.user_manager.last_recognized_user.user_id if self.user_manager.last_recognized_user else 'visitor'
                user_profile = None
                if current_user_id != 'visitor' and self.user_manager:
                    user_profile = self.user_manager.get_user_profile(current_user_id)
                    
                for chunk in self.dialogue_manager.generate_response_stream(
                    prompt=text,
                    user_id=current_user_id,
                    user_profile=user_profile
                ):
                    yield chunk
                self.system_state = SystemState.SPEAKING
                self.last_response_time = datetime.now()
                return
            if audio:
                self.system_state = SystemState.THINKING
                audio_text = self.voice_recognizer.recognize(audio)
                if audio_text:
                    logger.info(f"识别到语音: {audio_text}")
                    
                    # 获取声纹特征和用户信息
                    voice_feature = self.voice_recognizer.extract_voice_feature(np.frombuffer(audio, dtype=np.int16).astype(np.float32))
                    user_id, voice_conf = self.voice_recognizer.match_voice(voice_feature) if voice_feature is not None else (None, 0.0)
                    
                    # 获取用户画像信息
                    user_profile = None
                    if user_id and user_id != 'unknown' and self.user_manager:
                        user_profile = self.user_manager.get_user_profile(user_id)
                        if user_profile:
                            logger.info(f"👤 用户画像: {user_profile['name']} (访问{user_profile['visit_count']}次)")
                    
                    # 流式生成个性化回复
                    for chunk in self.dialogue_manager.generate_response_stream(
                        audio_text=audio_text,
                        user_id=user_id,
                        user_profile=user_profile
                    ):
                        yield chunk
                    self.system_state = SystemState.SPEAKING
                    self.last_response_time = datetime.now()
                    return
                else:
                    logger.debug("未识别到语音内容")
            if image:
                self.system_state = SystemState.THINKING
                try:
                    nparr = np.frombuffer(image, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        face_info = self.face_recognizer.detect(frame)
                        if face_info and face_info.get("detected"):
                            logger.info("👤 检测到人脸")
                            logger.debug("已记录视觉信息，等待用户语音输入")
                            # 不调用大模型，只记录视觉信息
                except Exception as e:
                    logger.error(f"图像处理失败: {e}")
            self.system_state = SystemState.IDLE
            yield None
        except Exception as e:
            logger.error(f"流式处理输入失败: {e}")
            self.system_state = SystemState.ERROR
            yield None 

    # 画像加密存储、删除接口
    def save_user_profiles(self, encrypt=True):
        if self.user_manager:
            data = self.user_manager.export_profiles()
            if encrypt:
                data = encrypt_data(data)
            with open('user_profiles.enc', 'wb') as f:
                f.write(data)
            logger.info("用户画像已加密保存")
    def load_user_profiles(self, decrypt=True):
        if self.user_manager:
            with open('user_profiles.enc', 'rb') as f:
                data = f.read()
            if decrypt:
                data = decrypt_data(data)
            self.user_manager.import_profiles(data)
            logger.info("用户画像已解密加载")
    def delete_user_profile(self, user_id):
        if self.user_manager:
            self.user_manager.delete_user_profile(user_id)
            logger.info(f"已删除用户画像: {user_id}")
    # 配置热加载、模块热插拔接口（预留）
    def reload_config(self):
        logger.info("配置热加载功能预留")
    def reload_module(self, module_name):
        logger.info(f"模块热插拔功能预留: {module_name}") 