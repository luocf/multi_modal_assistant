#!/usr/bin/env python3
"""测试音频采集功能"""

import pyaudio
import numpy as np
import time
from loguru import logger

class AudioCapture:
    def __init__(self, config):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = []
        self.silence_frames = 0
        self.max_silence_frames = int(config.get("silence_limit", 1.0) * config["rate"] / config["chunk"])
        
    def start(self):
        """启动音频采集"""
        if self.stream is None:
            self.stream = self.audio.open(
                format=self.config["format"],
                channels=self.config["channels"],
                rate=self.config["rate"],
                input=True,
                frames_per_buffer=self.config["chunk"]
            )
        self.is_recording = True
        self.frames = []
        self.silence_frames = 0
        
    def stop(self):
        """停止音频采集"""
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"关闭音频流失败: {e}")
            finally:
                self.stream = None
            
    def get_audio_data(self):
        """获取音频数据"""
        if not self.is_recording or self.stream is None:
            return np.array([])
            
        try:
            data = self.stream.read(self.config["chunk"], exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # 检测音量
            volume = np.abs(audio_data).mean()
            if volume < self.config.get("threshold", 100):
                self.silence_frames += 1
            else:
                self.silence_frames = 0
                
            return audio_data
            
        except Exception as e:
            logger.error(f"读取音频数据失败: {e}")
            return np.array([])
            
    def __del__(self):
        """清理资源"""
        self.stop()
        if self.audio:
            self.audio.terminate()

def test_audio_capture():
    """测试音频采集"""
    logger.info("开始测试音频采集...")
    
    # 音频配置
    audio_config = {
        "format": pyaudio.paInt16,
        "channels": 1,
        "rate": 16000,
        "chunk": 1024,
        "threshold": 100,
        "silence_limit": 1.0
    }
    
    # 创建音频采集器
    audio_capture = AudioCapture(audio_config)
    
    try:
        # 启动音频采集
        audio_capture.start()
        logger.info("音频采集已启动，开始录音测试（10秒）...")
        
        start_time = time.time()
        frame_count = 0
        total_volume = 0
        
        while time.time() - start_time < 10:  # 测试10秒
            audio_data = audio_capture.get_audio_data()
            if len(audio_data) > 0:
                frame_count += 1
                volume = np.abs(audio_data).mean()
                total_volume += volume
                
                # 每秒显示一次统计
                if frame_count % 16 == 0:  # 大约每秒（16000/1024 ≈ 16帧）
                    avg_volume = total_volume / frame_count
                    logger.info(f"帧数: {frame_count}, 当前音量: {volume:.2f}, 平均音量: {avg_volume:.2f}")
            
            time.sleep(0.01)  # 短暂休息
        
        logger.info(f"测试完成！总帧数: {frame_count}, 平均音量: {total_volume/frame_count:.2f}")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
    finally:
        audio_capture.stop()

if __name__ == "__main__":
    test_audio_capture() 