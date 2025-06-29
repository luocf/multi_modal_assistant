from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).parent.parent.parent

# 模型配置
MODEL_CONFIG: Dict[str, Any] = {
    "default_model": "qwen_cloud",
    "models": {
        "qwen": {
            "name": "Qwen/Qwen-1.5B-Chat",
            "type": "qwen",
            "size": "1.5B",
            "device": "auto"
        }
    }
}

# 摄像头配置
CAMERA_CONFIG = {
    "width": 640,
    "height": 480,
    "fps": 30
}

# 音频配置
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024
}

# 识别阈值
RECOGNITION_THRESHOLDS = {
    "face_similarity": 0.6,
    "voice_similarity": 0.7,
    "emotion_confidence": 0.5,
    "gesture_confidence": 0.6
}

# 用户画像配置
PROFILE_CONFIG = {
    "max_history_length": 100,
    "emotion_history_size": 50,
    "dialogue_summary_size": 20
}

# 对话配置
DIALOGUE_CONFIG = {
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9
}

# 存储路径
STORAGE_PATHS = {
    "user_profiles": BASE_DIR / "data/profiles",
    "face_embeddings": BASE_DIR / "data/faces",
    "voice_embeddings": BASE_DIR / "data/voices",
    "logs": BASE_DIR / "logs"
}

# 创建必要的目录
for path in STORAGE_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# API密钥（从环境变量获取）
API_KEYS = {
    "modelscope": os.getenv("MODELSCOPE_API_KEY", ""),
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "rotation": "1 day",
    "retention": "7 days"
} 