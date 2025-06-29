"""用户管理器配置"""

USER_CONFIG = {
    # 用户状态配置
    "inactive_timeout": 300,  # 用户不活跃超时时间（秒）
    "max_inactive_time": 3600,  # 最大不活跃时间（秒）
    
    # 用户识别配置
    "face_threshold": 0.6,  # 人脸识别阈值
    "voice_threshold": 0.75,  # 声纹识别阈值
    "fusion_threshold": 0.7,  # 多模态融合阈值
    
    # 用户数据配置
    "max_face_embeddings": 10,  # 最大人脸特征数
    "max_embeddings_per_user": 5,  # 每个用户最多存储的人脸特征数
    "max_voice_samples": 10,  # 最大声纹样本数
    "min_voice_samples": 3,  # 最小声纹样本数
    
    # 特征更新配置
    "high_similarity_threshold": 0.8,  # 高相似度阈值
    "high_similarity_weight": 0.7,  # 高相似度权重
    "low_similarity_weight": 0.3,  # 低相似度权重
    
    # 多模态融合配置
    "face_weight_ratio": 0.6,  # 人脸权重比例
    "voice_weight_ratio": 0.4,  # 声纹权重比例
    
    # 日志配置
    "log_level": "INFO",  # 日志级别
    "log_file": "logs/user_manager.log",  # 日志文件路径
} 