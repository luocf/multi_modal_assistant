from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import threading
import time
import os
from collections import deque
from queue import Queue
import pyaudio
from loguru import logger
from speechbrain.inference import EncoderClassifier
from app.core.utils import compute_cosine_similarity

class VoiceRecognizer:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化声纹识别系统
        Args:
            config: 配置字典，包含：
                - sample_rate: 音频采样率
                - chunk_size: 音频块大小
                - voice_threshold: 声纹匹配阈值
        """
        self.config = config
        self.logger = logger
        self.sample_rate = config.get("sample_rate", 16000)
        self.chunk_size = config.get("chunk_size", 1024)
        self.voice_threshold = config.get("voice_threshold", 0.75)
        
        # 初始化SpeechBrain声纹识别模型
        try:
            self.encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/spkrec-ecapa-voxceleb",
                run_opts={"device": "cpu"}
            )
            self.logger.info("声纹识别模型加载成功")
        except Exception as e:
            self.logger.error(f"声纹识别模型加载失败: {e}")
            raise
            
        # 音频处理相关
        self.audio_buffer = Queue()
        self.is_recording = False
        self.recording_thread = None
        self.processing_thread = None
        
        # 声纹特征缓存
        self.voice_embeddings = {}  # user_id -> embedding
        self.voice_profiles = {}  # user_id -> List[np.ndarray]
        self.recent_embeddings = deque(maxlen=5)  # 最近的5个声纹特征
        self.last_process_time = 0
        self.process_interval = 1.0  # 处理间隔（秒）
        
    def start_recording(self):
        """开始录音"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.processing_thread = threading.Thread(target=self._process_audio)
        
        self.recording_thread.start()
        self.processing_thread.start()
        self.logger.info("开始录音和声纹识别")
        
    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        if self.processing_thread:
            self.processing_thread.join()
        self.logger.info("停止录音和声纹识别")
        
    def _record_audio(self):
        """录音线程"""
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        while self.is_recording:
            try:
                data = stream.read(self.chunk_size)
                self.audio_buffer.put(data)
            except Exception as e:
                self.logger.error(f"录音错误: {e}")
                break
                
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
    def _process_audio(self):
        """音频处理线程"""
        buffer_size = int(self.sample_rate * 3)  # 3秒音频
        audio_buffer = []
        
        while self.is_recording:
            try:
                # 获取音频数据
                if not self.audio_buffer.empty():
                    data = self.audio_buffer.get()
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    audio_buffer.extend(audio_data)
                    
                    # 当累积足够的音频数据时进行处理
                    if len(audio_buffer) >= buffer_size:
                        current_time = time.time()
                        
                        # 控制处理频率
                        if current_time - self.last_process_time >= self.process_interval:
                            # 转换为tensor
                            audio_tensor = torch.FloatTensor(audio_buffer[:buffer_size])
                            
                            # 提取声纹特征
                            with torch.no_grad():
                                embedding = self.encoder.encode_batch(audio_tensor.unsqueeze(0))
                                embedding = embedding.squeeze().numpy()
                                
                            # 添加到最近的特征列表
                            self.recent_embeddings.append(embedding)
                            
                            # 更新时间戳
                            self.last_process_time = current_time
                            
                        # 清理缓冲区，保留最后1秒的数据用于下次处理
                        audio_buffer = audio_buffer[-self.sample_rate:]
                        
            except Exception as e:
                self.logger.error(f"音频处理错误: {e}")
                continue
                
    def get_current_voice_embedding(self) -> Optional[np.ndarray]:
        """获取当前的声纹特征"""
        if not self.recent_embeddings:
            return None
        return np.mean([emb for emb in self.recent_embeddings], axis=0)
        
    def register_voice(self, user_id: str, voice_embedding: np.ndarray):
        """注册声纹"""
        self.voice_embeddings[user_id] = voice_embedding
        self.logger.info(f"注册声纹: {user_id}")
        
    def match_voice(self, voice_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        匹配声纹
        Returns:
            (user_id, confidence) 或 (None, 0.0)
        """
        if not self.voice_embeddings or voice_embedding is None:
            return None, 0.0
            
        try:
            # 计算与所有已知声纹的相似度
            similarities = {}
            for user_id, known_embedding in self.voice_embeddings.items():
                similarity = compute_cosine_similarity(voice_embedding, known_embedding)
                similarities[user_id] = similarity
                
            # 找到最相似的用户
            best_match = max(similarities.items(), key=lambda x: x[1])
            
            if best_match[1] >= self.voice_threshold:
                return best_match[0], best_match[1]
                
            return None, best_match[1]
            
        except Exception as e:
            self.logger.error(f"声纹匹配错误: {e}")
            return None, 0.0
            
    def extract_voice_feature(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """提取声纹特征"""
        try:
            if len(audio_data) < self.sample_rate:  # 少于1秒的音频
                return None
                
            # 预处理音频
            audio_tensor = self._preprocess_audio(audio_data)
            
            # 提取特征
            with torch.no_grad():
                embedding = self.encoder.encode_batch(audio_tensor.unsqueeze(0))
                embedding = embedding.squeeze().numpy()
                
            return embedding
            
        except Exception as e:
            self.logger.error(f"声纹特征提取失败: {e}")
            return None
            
    def recognize(self, audio_data: bytes) -> str:
        """
        语音识别（语音转文字）
        注意：这是一个简化的实现，实际项目中应该使用专门的ASR模型
        
        Args:
            audio_data: 音频数据（bytes格式）
            
        Returns:
            识别出的文字
        """
        try:
            # 将bytes转换为numpy数组
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            else:
                audio_array = audio_data
                
            # 检查音频长度
            if len(audio_array) < self.sample_rate * 0.5:  # 少于0.5秒
                self.logger.debug("音频太短，无法识别")
                return ""
                
            # 计算音频能量来判断是否有语音
            energy = np.sum(audio_array ** 2) / len(audio_array)
            self.logger.debug(f"音频能量: {energy}")
            
            if energy > 0.001:  # 有语音能量
                # 简化的语音识别逻辑
                # 在实际应用中，这里应该调用专门的ASR模型
                
                # 根据音频特征返回不同的模拟识别结果
                duration = len(audio_array) / self.sample_rate
                
                if duration < 1.0:
                    return "嗯"
                elif duration < 2.0:
                    return "你好"
                elif duration < 3.0:
                    return "今天天气怎么样"
                else:
                    return "我想了解一些信息"
            else:
                self.logger.debug("未检测到语音能量")
                return ""
                
        except Exception as e:
            self.logger.error(f"语音识别失败: {e}")
            return ""
            
    def _preprocess_audio(self, audio_data: np.ndarray) -> torch.Tensor:
        """预处理音频数据"""
        try:
            # 归一化
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # 转换为tensor
            audio_tensor = torch.FloatTensor(audio_data)
            
            return audio_tensor
            
        except Exception as e:
            self.logger.error(f"音频预处理失败: {e}")
            return torch.FloatTensor(audio_data)
            
    def update_voice_profile(self, user_id: str, voice_feature: np.ndarray, audio_data: Optional[np.ndarray] = None):
        """更新用户声纹档案"""
        try:
            if user_id not in self.voice_profiles:
                self.voice_profiles[user_id] = []
                
            # 质量评估
            if audio_data is not None:
                quality_score = self._evaluate_voice_quality(audio_data)
                if quality_score.get("overall_quality", 0) < 0.6:
                    self.logger.debug(f"声纹质量较低，跳过更新: {user_id}")
                    return
                    
            # 增量学习
            existing_features = self.voice_profiles[user_id]
            if existing_features:
                updated_feature = self._incremental_learning(existing_features, voice_feature)
                self.voice_embeddings[user_id] = updated_feature
            else:
                self.voice_embeddings[user_id] = voice_feature
                
            # 添加到档案
            self.voice_profiles[user_id].append(voice_feature)
            
            # 限制档案大小
            max_samples = self.config.get("max_voice_samples", 10)
            if len(self.voice_profiles[user_id]) > max_samples:
                self.voice_profiles[user_id] = self.voice_profiles[user_id][-max_samples:]
                
            self.logger.info(f"更新声纹档案: {user_id}, 样本数: {len(self.voice_profiles[user_id])}")
            
        except Exception as e:
            self.logger.error(f"更新声纹档案失败: {e}")
            
    def _incremental_learning(self, existing_features: List[np.ndarray], new_feature: np.ndarray) -> np.ndarray:
        """增量学习更新声纹特征"""
        try:
            # 计算新特征与现有特征的相似度
            similarities = [compute_cosine_similarity(new_feature, feat) for feat in existing_features]
            max_similarity = max(similarities) if similarities else 0
            
            # 根据相似度调整权重
            high_sim_threshold = self.config.get("high_similarity_threshold", 0.8)
            if max_similarity > high_sim_threshold:
                # 高相似度，较小权重更新
                weight = self.config.get("high_similarity_weight", 0.7)
            else:
                # 低相似度，较大权重更新
                weight = self.config.get("low_similarity_weight", 0.3)
                
            # 计算现有特征的平均值
            mean_existing = np.mean(existing_features, axis=0)
            
            # 加权平均更新
            updated_feature = weight * mean_existing + (1 - weight) * new_feature
            
            return updated_feature
            
        except Exception as e:
            self.logger.error(f"增量学习失败: {e}")
            return new_feature
            
    def _evaluate_voice_quality(self, audio_data: np.ndarray) -> Dict[str, float]:
        """评估声音质量"""
        try:
            quality_scores = {}
            
            # 信噪比
            quality_scores["snr"] = self._calculate_snr(audio_data)
            
            # 音量
            quality_scores["volume"] = self._calculate_volume(audio_data)
            
            # 语音活动检测
            quality_scores["vad"] = self._calculate_vad_score(audio_data)
            
            # 频谱质量
            quality_scores["spectral"] = self._calculate_spectral_quality(audio_data)
            
            # 综合质量分数
            weights = {"snr": 0.3, "volume": 0.2, "vad": 0.3, "spectral": 0.2}
            overall_quality = sum(quality_scores[key] * weights[key] for key in weights.keys())
            quality_scores["overall_quality"] = overall_quality
            
            return quality_scores
            
        except Exception as e:
            self.logger.error(f"质量评估失败: {e}")
            return {"overall_quality": 0.5}
            
    def _calculate_snr(self, audio_data: np.ndarray) -> float:
        """计算信噪比"""
        try:
            # 简化的SNR计算
            signal_power = np.var(audio_data)
            noise_estimate = np.var(audio_data[:int(len(audio_data) * 0.1)])  # 前10%作为噪声估计
            
            if noise_estimate > 0:
                snr = 10 * np.log10(signal_power / noise_estimate)
                return min(max(snr / 20, 0), 1)  # 归一化到0-1
            return 0.5
            
        except Exception:
            return 0.5
            
    def _calculate_volume(self, audio_data: np.ndarray) -> float:
        """计算音量"""
        try:
            volume = np.sqrt(np.mean(audio_data ** 2))
            return min(volume * 10, 1)  # 归一化
        except Exception:
            return 0.5
            
    def _calculate_vad_score(self, audio_data: np.ndarray) -> float:
        """计算语音活动检测分数"""
        try:
            # 简化的VAD，基于能量
            frame_size = 512
            frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data), frame_size)]
            
            energy_threshold = np.mean([np.sum(frame**2) for frame in frames]) * 0.5
            voice_frames = sum(1 for frame in frames if np.sum(frame**2) > energy_threshold)
            
            return voice_frames / len(frames) if frames else 0
            
        except Exception:
            return 0.5
            
    def _calculate_spectral_quality(self, audio_data: np.ndarray) -> float:
        """计算频谱质量"""
        try:
            # 简化的频谱质量评估
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft)
            
            # 计算频谱平坦度
            spectral_flatness = np.exp(np.mean(np.log(magnitude + 1e-10))) / np.mean(magnitude)
            
            return min(spectral_flatness * 2, 1)
            
        except Exception:
            return 0.5
            
    def save_voice_embeddings(self, save_path: str = "models/voice_embeddings.npz"):
        """保存声纹特征"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(
                save_path,
                embeddings=self.voice_embeddings,
                profiles=self.voice_profiles,
                metadata={"last_update": time.time()}
            )
            self.logger.info(f"声纹特征保存成功: {save_path}")
        except Exception as e:
            self.logger.error(f"声纹特征保存失败: {e}")
            
    def load_voice_embeddings(self, load_path: str = "models/voice_embeddings.npz"):
        """加载声纹特征"""
        try:
            if os.path.exists(load_path):
                data = np.load(load_path, allow_pickle=True)
                self.voice_embeddings = data["embeddings"].item()
                if "profiles" in data:
                    self.voice_profiles = data["profiles"].item()
                self.logger.info(f"声纹特征加载成功: {len(self.voice_embeddings)} 个用户")
            else:
                self.logger.warning(f"声纹特征文件不存在: {load_path}")
        except Exception as e:
            self.logger.error(f"声纹特征加载失败: {e}")
            
    def get_voice_profile(self, user_id: str) -> List[np.ndarray]:
        """获取用户声纹档案"""
        return self.voice_profiles.get(user_id, [])
        
    def remove_voice_profile(self, user_id: str):
        """删除用户声纹档案"""
        if user_id in self.voice_embeddings:
            del self.voice_embeddings[user_id]
        if user_id in self.voice_profiles:
            del self.voice_profiles[user_id]
        self.logger.info(f"删除声纹档案: {user_id}")
        
    def reset(self):
        """重置声纹识别器状态"""
        self.stop_recording()
        self.recent_embeddings.clear()
        self.last_process_time = 0
        
    def close(self):
        """关闭声纹识别器"""
        self.stop_recording()
        self.save_voice_embeddings()
        self.logger.info("声纹识别器已关闭") 