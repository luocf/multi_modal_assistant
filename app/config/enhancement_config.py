"""增强器配置"""

AUDIO_ENHANCER_CONFIG = {
    "sample_rate": 16000,
    "frame_length": 512,
    "hop_length": 128,
    "min_volume": -60,
    "target_volume": -20,
    "noise_reduce_amount": 0.75,
    "vad_mode": 3,  # WebRTC VAD的模式 (0-3)
    "use_gpu": True
}

VIDEO_ENHANCER_CONFIG = {
    "target_fps": 30,
    "min_face_size": 64,
    "quality_threshold": 0.7,
    "brightness_range": (0.4, 0.6),
    "contrast_range": (0.4, 0.6),
    "use_gpu": True
}

MULTIMODAL_FUSION_CONFIG = {
    "feature_dims": {
        "face": 512,
        "voice": 512,
        "lip": 512,
        "behavior": 512
    },
    "time_window": 1.0,
    "min_confidence": 0.5,
    "use_gpu": True
}

# 性能监控配置
PERFORMANCE_MONITOR_CONFIG = {
    "metrics_window": 60,  # 60秒的统计窗口
    "alert_thresholds": {
        "cpu_percent": 80,
        "memory_percent": 80,
        "gpu_memory_percent": 80,
        "latency_ms": 100
    },
    "log_interval": 5,  # 每5秒记录一次
    "enable_gpu_monitoring": True
}

# 异常处理配置
ERROR_HANDLING_CONFIG = {
    "max_retries": 3,        # 最大重试次数
    "retry_interval": 0.5,   # 重试间隔（秒）
    "fallback_modes": {
        "face_only": True,   # 是否允许仅人脸模式
        "voice_only": True,  # 是否允许仅声纹模式
        "minimum_modes": 2   # 最少需要的模态数
    }
}

# 日志配置
LOGGING_CONFIG = {
    "log_level": "INFO",
    "file_path": "logs/enhancement.log",
    "rotation": "1 day",
    "retention": "1 week",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
} 