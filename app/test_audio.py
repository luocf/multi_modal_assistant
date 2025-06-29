#!/usr/bin/env python3
"""音频测试工具"""

import pyaudio
import numpy as np
import wave
import time
from loguru import logger

def test_audio_input():
    """测试音频输入"""
    logger.info("开始测试音频输入...")
    
    # 音频配置
    config = {
        "format": pyaudio.paInt16,
        "channels": 1,
        "rate": 16000,
        "chunk": 1024
    }
    
    audio = pyaudio.PyAudio()
    
    try:
        # 列出所有音频设备
        logger.info("可用的音频设备:")
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            logger.info(f"设备 {i}: {info['name']} - 输入通道: {info['maxInputChannels']}")
        
        # 打开音频流
        stream = audio.open(
            format=config["format"],
            channels=config["channels"],
            rate=config["rate"],
            input=True,
            frames_per_buffer=config["chunk"]
        )
        
        logger.info("开始录音测试，持续5秒...")
        frames = []
        
        for i in range(0, int(config["rate"] / config["chunk"] * 5)):
            data = stream.read(config["chunk"])
            frames.append(data)
            
            # 计算音量
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            max_volume = np.abs(audio_data).max()
            
            if i % 10 == 0:  # 每秒显示一次
                logger.info(f"音量: 平均={volume:.2f}, 最大={max_volume:.2f}")
        
        logger.info("录音完成")
        
        # 保存音频文件
        with wave.open("test_audio.wav", "wb") as wf:
            wf.setnchannels(config["channels"])
            wf.setsampwidth(audio.get_sample_size(config["format"]))
            wf.setframerate(config["rate"])
            wf.writeframes(b''.join(frames))
        
        logger.info("音频已保存到 test_audio.wav")
        
        # 计算总体统计
        all_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        total_volume = np.abs(all_data).mean()
        total_max = np.abs(all_data).max()
        
        logger.info(f"总体统计: 平均音量={total_volume:.2f}, 最大音量={total_max:.2f}")
        
        if total_volume < 10:
            logger.warning("音量过低，可能麦克风未正常工作")
        elif total_volume > 1000:
            logger.warning("音量过高，可能存在噪音")
        else:
            logger.info("音量正常")
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        logger.error(f"音频测试失败: {e}")
    finally:
        audio.terminate()

if __name__ == "__main__":
    test_audio_input() 