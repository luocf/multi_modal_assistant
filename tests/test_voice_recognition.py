import sys
import os
import time
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.perception.audio_capture import AudioCapture
from core.perception.voice_recognizer import VoiceRecognizer
from config.voice_config import VOICE_CONFIG

def test_voice_recognition():
    """测试声纹识别功能"""
    print("初始化声纹识别器...")
    voice_recognizer = VoiceRecognizer(VOICE_CONFIG)
    
    print("初始化音频采集器...")
    audio_capture = AudioCapture(VOICE_CONFIG)
    
    try:
        # 开始录音
        print("开始录音，请说话...")
        audio_capture.start()
        time.sleep(5)  # 录制5秒
        audio_data = audio_capture.record(5)
        
        if len(audio_data) == 0:
            print("未检测到音频数据")
            return
            
        # 评估音频质量
        quality_metrics = voice_recognizer._evaluate_voice_quality(audio_data)
        print("\n音频质量评估结果:")
        print(f"总体质量分数: {quality_metrics['quality_score']:.2f}")
        print(f"信噪比: {quality_metrics['snr']:.2f}")
        print(f"音量: {quality_metrics['volume']:.2f}")
        print(f"语音活动检测分数: {quality_metrics['vad_score']:.2f}")
        print(f"频谱质量: {quality_metrics['spectral_quality']:.2f}")
        
        # 提取声纹特征
        print("\n提取声纹特征...")
        voice_feature = voice_recognizer.extract_voice_feature(audio_data)
        
        if voice_feature is None:
            print("声纹特征提取失败")
            return
            
        print(f"声纹特征维度: {voice_feature.shape}")
        
        # 测试声纹匹配
        print("\n测试声纹匹配...")
        user_id, similarity = voice_recognizer.match_voice(voice_feature)
        
        if user_id:
            print(f"匹配到用户: {user_id}, 相似度: {similarity:.2f}")
        else:
            print("未匹配到用户")
            
        # 保存音频数据
        print("\n保存音频数据...")
        audio_capture.save_audio(audio_data, "test_audio.wav")
        print("音频数据已保存到 test_audio.wav")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        
    finally:
        # 清理资源
        audio_capture.stop()
        print("\n测试完成")

if __name__ == "__main__":
    test_voice_recognition() 