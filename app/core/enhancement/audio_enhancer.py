import numpy as np
from typing import Dict, Any, Optional, Tuple
import scipy.signal as signal
from collections import deque
import time
from loguru import logger
import webrtcvad
import torch
import torchaudio
import torchaudio.transforms as T

class AudioEnhancer:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化音频增强器
        Args:
            config: 配置字典，包含：
                - sample_rate: 采样率
                - frame_length: 帧长度
                - noise_reduce_threshold: 噪声抑制阈值
                - vad_mode: VAD模式(0-3)
                - max_history_seconds: 历史缓存时长
        """
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        self.frame_length = config.get("frame_length", 512)
        self.noise_threshold = config.get("noise_reduce_threshold", 0.1)
        
        # 初始化VAD
        self.vad = webrtcvad.Vad(config.get("vad_mode", 3))
        
        # 初始化降噪模型
        self.denoiser = None
        if torch.cuda.is_available():
            try:
                self.denoiser = torchaudio.pipelines.DN3(download=True)
                self.denoiser.to("cuda")
                logger.info("使用GPU进行音频降噪")
            except Exception as e:
                logger.warning(f"加载降噪模型失败: {e}")
                
        # 音频统计
        self.audio_stats = {
            "volume_history": deque(maxlen=100),
            "noise_floor": None,
            "peak_volume": 0.0,
            "last_active_time": 0.0
        }
        
        # 自适应参数
        self.adaptive_params = {
            "noise_threshold": self.noise_threshold,
            "vad_threshold": 0.5,
            "gain": 1.0
        }
        
        logger.info("音频增强器初始化完成")
        
    def enhance(self, audio_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        增强音频质量
        Returns:
            (enhanced_audio, stats)
        """
        try:
            # 1. 更新音频统计
            self._update_stats(audio_data)
            
            # 2. 自适应参数调整
            self._adapt_parameters()
            
            # 3. 应用增强
            enhanced = audio_data.copy()
            
            # 3.1 音量归一化
            if self.adaptive_params["gain"] != 1.0:
                enhanced = enhanced * self.adaptive_params["gain"]
                
            # 3.2 降噪
            if self.denoiser is not None:
                try:
                    # 转换为tensor
                    audio_tensor = torch.from_numpy(enhanced).float()
                    if torch.cuda.is_available():
                        audio_tensor = audio_tensor.cuda()
                        
                    # 应用降噪
                    with torch.no_grad():
                        denoised = self.denoiser(audio_tensor)
                        enhanced = denoised.cpu().numpy()
                except Exception as e:
                    logger.error(f"降噪处理失败: {e}")
                    
            # 3.3 频谱抑制
            if self.audio_stats["noise_floor"] is not None:
                enhanced = self._spectral_gating(enhanced)
                
            # 4. 生成统计信息
            stats = {
                "original_volume": np.abs(audio_data).mean(),
                "enhanced_volume": np.abs(enhanced).mean(),
                "noise_floor": self.audio_stats["noise_floor"],
                "peak_volume": self.audio_stats["peak_volume"],
                "is_speech": self._is_speech(enhanced),
                "gain_applied": self.adaptive_params["gain"]
            }
            
            return enhanced, stats
            
        except Exception as e:
            logger.error(f"音频增强失败: {e}")
            return audio_data, {}
            
    def _update_stats(self, audio_data: np.ndarray):
        """更新音频统计信息"""
        try:
            # 计算音量
            volume = np.abs(audio_data).mean()
            self.audio_stats["volume_history"].append(volume)
            
            # 更新峰值音量
            peak = np.abs(audio_data).max()
            self.audio_stats["peak_volume"] = max(self.audio_stats["peak_volume"], peak)
            
            # 更新噪声基准
            if len(self.audio_stats["volume_history"]) >= 50:  # 至少需要50帧
                sorted_volumes = sorted(self.audio_stats["volume_history"])
                # 使用前10%的音量作为噪声基准
                noise_floor = np.mean(sorted_volumes[:len(sorted_volumes)//10])
                self.audio_stats["noise_floor"] = noise_floor
                
        except Exception as e:
            logger.error(f"更新音频统计失败: {e}")
            
    def _adapt_parameters(self):
        """自适应参数调整"""
        try:
            if self.audio_stats["noise_floor"] is not None:
                # 调整噪声阈值
                self.adaptive_params["noise_threshold"] = max(
                    self.noise_threshold,
                    self.audio_stats["noise_floor"] * 1.5
                )
                
                # 调整音量增益
                if self.audio_stats["peak_volume"] > 0:
                    target_peak = 0.8  # 目标峰值
                    current_peak = self.audio_stats["peak_volume"]
                    if current_peak < target_peak / 2:
                        # 音量过低，增加增益
                        self.adaptive_params["gain"] = min(2.0, self.adaptive_params["gain"] * 1.1)
                    elif current_peak > target_peak:
                        # 音量过高，降低增益
                        self.adaptive_params["gain"] = max(0.5, self.adaptive_params["gain"] * 0.9)
                        
        except Exception as e:
            logger.error(f"参数自适应调整失败: {e}")
            
    def _spectral_gating(self, audio_data: np.ndarray) -> np.ndarray:
        """频谱抑制降噪"""
        try:
            # 计算STFT
            f, t, Zxx = signal.stft(
                audio_data,
                fs=self.sample_rate,
                nperseg=self.frame_length,
                noverlap=self.frame_length // 2
            )
            
            # 计算幅度谱
            mag = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # 应用频谱抑制
            noise_floor = self.audio_stats["noise_floor"]
            if noise_floor is not None:
                # 计算抑制因子
                gain = np.maximum(0, 1 - (noise_floor * 2 / (mag + 1e-6)))
                mag = mag * gain
                
            # 重建信号
            Zxx_enhanced = mag * np.exp(1j * phase)
            _, enhanced = signal.istft(
                Zxx_enhanced,
                fs=self.sample_rate,
                nperseg=self.frame_length,
                noverlap=self.frame_length // 2
            )
            
            return enhanced
            
        except Exception as e:
            logger.error(f"频谱抑制失败: {e}")
            return audio_data
            
    def _is_speech(self, audio_data: np.ndarray) -> bool:
        """检测是否为语音"""
        try:
            # 确保数据类型正确
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
                
            # 分帧处理
            frame_length = int(self.sample_rate * 0.03)  # 30ms
            frames = []
            for i in range(0, len(audio_data) - frame_length, frame_length):
                frames.append(audio_data[i:i + frame_length])
                
            # VAD检测
            speech_frames = 0
            for frame in frames:
                try:
                    if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                        speech_frames += 1
                except Exception:
                    continue
                    
            # 如果超过30%的帧被检测为语音，则认为是语音
            return speech_frames / len(frames) > 0.3 if frames else False
            
        except Exception as e:
            logger.error(f"语音检测失败: {e}")
            return False
            
    def reset_stats(self):
        """重置统计信息"""
        self.audio_stats = {
            "volume_history": deque(maxlen=100),
            "noise_floor": None,
            "peak_volume": 0.0,
            "last_active_time": 0.0
        }
        self.adaptive_params = {
            "noise_threshold": self.noise_threshold,
            "vad_threshold": 0.5,
            "gain": 1.0
        } 