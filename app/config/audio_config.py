"""音频配置"""

AUDIO_CONFIG = {
    "format": "int16",  # 音频格式
    "channels": 1,      # 声道数
    "rate": 16000,      # 采样率
    "chunk": 1024,      # 缓冲区大小
    "threshold": 0.1,   # 音量阈值
    "timeout": 5,       # 录音超时时间（秒）
    "silence_limit": 1  # 静音检测时间（秒）
} 