import pyaudio
import numpy as np
import wave
import threading
import queue
import time
from typing import Optional, Callable, Dict, Any
import logging
from pathlib import Path

class AudioCapture:
    def __init__(self, config: Dict[str, Any]):
        """初始化音频采集器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 音频参数
        self.sample_rate = config.get("sample_rate", 16000)  # 采样率
        self.chunk_size = config.get("chunk_size", 1024)     # 数据块大小
        self.channels = config.get("channels", 1)            # 声道数
        self.format = pyaudio.paFloat32                      # 采样格式
        
        # 音频流
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        
        # 数据队列
        self.audio_queue = queue.Queue()
        
        # 控制标志
        self.is_running = False
        self.is_recording = False
        
        # 回调函数
        self.on_audio_callback: Optional[Callable] = None
        
    def start(self):
        """启动音频采集，增加异常自恢复"""
        if self.is_running:
            return
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            self.is_running = True
            self.logger.info("音频采集启动成功")
        except Exception as e:
            self.logger.error(f"音频采集启动失败: {e}")
            # 尝试自恢复
            time.sleep(1)
            self.logger.info("尝试重新启动音频采集...")
            try:
                self.stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback
                )
                self.is_running = True
                self.logger.info("音频采集自恢复成功")
            except Exception as e2:
                self.logger.error(f"音频采集自恢复失败: {e2}")
                raise
            
    def stop(self):
        """停止音频采集"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.logger.info("音频采集已停止")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数
        
        Args:
            in_data: 输入音频数据
            frame_count: 帧数
            time_info: 时间信息
            status: 状态标志
            
        Returns:
            (None, pyaudio.paContinue)
        """
        if self.is_running:
            # 将字节数据转换为numpy数组
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # 将数据放入队列
            self.audio_queue.put(audio_data)
            
            # 如果设置了回调函数，则调用
            if self.on_audio_callback:
                self.on_audio_callback(audio_data)
                
        return (None, pyaudio.paContinue)
        
    def get_audio_data(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """获取音频数据
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            音频数据，如果超时则返回None
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def set_callback(self, callback: Callable[[np.ndarray], None]):
        """设置音频回调函数
        
        Args:
            callback: 回调函数，接收音频数据作为参数
        """
        self.on_audio_callback = callback
        
    def save_audio(self, audio_data: np.ndarray, filename: str):
        """保存音频数据到文件
        
        Args:
            audio_data: 音频数据
            filename: 文件名
        """
        try:
            # 确保目录存在
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为WAV文件
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paFloat32))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
                
            self.logger.info(f"音频数据已保存到: {filename}")
            
        except Exception as e:
            self.logger.error(f"保存音频数据失败: {e}")
            raise
            
    def record(self, duration: float, filename: Optional[str] = None) -> np.ndarray:
        """录制指定时长的音频
        
        Args:
            duration: 录制时长（秒）
            filename: 保存的文件名（可选）
            
        Returns:
            录制的音频数据
        """
        if not self.is_running:
            self.start()
            
        self.is_recording = True
        audio_data = []
        start_time = time.time()
        
        while time.time() - start_time < duration and self.is_recording:
            data = self.get_audio_data()
            if data is not None:
                audio_data.append(data)
                
        self.is_recording = False
        
        # 合并音频数据
        if audio_data:
            audio_data = np.concatenate(audio_data)
            
            # 如果指定了文件名，则保存
            if filename:
                self.save_audio(audio_data, filename)
                
            return audio_data
            
        return np.array([])
        
    def __del__(self):
        """析构函数"""
        self.stop()

    def set_format(self, fmt):
        """设置采样格式（如 pyaudio.paInt16, pyaudio.paFloat32）"""
        self.format = fmt
        self.logger.info(f"音频采样格式已设置为: {fmt}")

    def reset(self):
        """重置音频采集状态"""
        self.audio_queue.queue.clear()
        self.is_recording = False
        self.logger.info("音频采集器已重置")

    def close(self):
        """资源释放"""
        self.stop()
        self.reset()
        self.logger.info("音频采集器已关闭") 