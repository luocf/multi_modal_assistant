"""声纹识别配置"""

VOICE_CONFIG = {
    # 音频参数
    "sample_rate": 16000,  # 采样率
    "chunk_size": 1024,    # 数据块大小
    "channels": 1,         # 声道数
    
    # 声纹识别参数
    "voice_threshold": 0.75,      # 声纹匹配阈值
    "min_voice_samples": 3,       # 最小声纹样本数
    "max_voice_samples": 10,      # 最大声纹样本数
    
    # 质量评估参数
    "quality_threshold": 0.6,     # 质量分数阈值
    "snr_threshold": 0.5,         # 信噪比阈值
    "volume_threshold": 0.3,      # 音量阈值
    "vad_threshold": 0.4,         # 语音活动检测阈值
    "spectral_threshold": 0.4,    # 频谱质量阈值
    
    # 增量学习参数
    "high_similarity_threshold": 0.8,  # 高相似度阈值
    "high_similarity_weight": 0.7,     # 高相似度权重
    "low_similarity_weight": 0.3,      # 低相似度权重
    
    # 多模态融合参数
    "fusion_threshold": 0.7,          # 融合置信度阈值
    "face_weight_ratio": 0.6,         # 人脸权重比例
    "voice_weight_ratio": 0.4,        # 声纹权重比例
    
    # 模型参数
    "model_source": "speechbrain/spkrec-ecapa-voxceleb",  # 模型来源
    "model_savedir": "models/spkrec-ecapa-voxceleb",      # 模型保存目录
} 