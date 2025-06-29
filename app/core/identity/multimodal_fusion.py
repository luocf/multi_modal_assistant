import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time
from loguru import logger
from scipy.spatial.distance import cosine
from enum import Enum

class ModalityType(Enum):
    FACE = "face"
    VOICE = "voice"
    LIP = "lip"
    BEHAVIOR = "behavior"

@dataclass
class ModalityFeature:
    type: ModalityType
    feature: np.ndarray
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]

class FusionModel(nn.Module):
    def __init__(self, feature_dims: Dict[str, int]):
        super().__init__()
        
        # 特征转换层
        self.feature_transforms = nn.ModuleDict({
            modality_type: nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            for modality_type, dim in feature_dims.items()
        })
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=0.1
        )
        
        # 融合层
        self.fusion_layers = nn.Sequential(
            nn.Linear(256 * len(feature_dims), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. 特征转换
        transformed_features = {
            modality: self.feature_transforms[modality](feature)
            for modality, feature in features.items()
        }
        
        # 2. 堆叠特征用于注意力机制
        stacked_features = torch.stack(list(transformed_features.values()), dim=0)
        
        # 3. 应用注意力机制
        attended_features, _ = self.attention(
            stacked_features,
            stacked_features,
            stacked_features
        )
        
        # 4. 展平并连接特征
        flat_features = torch.cat([
            feat.mean(dim=0) for feat in attended_features.split(1, dim=0)
        ], dim=1)
        
        # 5. 最终融合
        fused_feature = self.fusion_layers(flat_features)
        
        return fused_feature

class MultiModalFusion:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化多模态融合系统
        Args:
            config: 配置字典，包含：
                - feature_dims: 各模态特征维度
                - time_window: 时间窗口大小(秒)
                - min_confidence: 最小置信度
                - use_gpu: 是否使用GPU
        """
        try:
            # 1. 基本配置
            self.feature_dims = {
                ModalityType.FACE: config["feature_dims"]["face"],
                ModalityType.VOICE: config["feature_dims"]["voice"],
                ModalityType.LIP: config["feature_dims"]["lip"],
                ModalityType.BEHAVIOR: config["feature_dims"]["behavior"]
            }
            self.time_window = config.get("time_window", 1.0)
            self.min_confidence = config.get("min_confidence", 0.5)
            self.min_modalities = config.get("min_modalities", 2)
            self.similarity_threshold = config.get("similarity_threshold", 0.8)
            
            # 2. 特征缓冲区
            self.feature_buffer: Dict[ModalityType, List[ModalityFeature]] = {
                modality_type: []
                for modality_type in ModalityType
            }
            
            # 3. 异常检测阈值
            self.anomaly_thresholds = {
                "time_diff": config.get("time_diff_threshold", 0.5),
                "min_modalities": self.min_modalities,
                "feature_diff": config.get("feature_diff_threshold", 0.3)
            }
            
            # 4. 初始化设备
            self.device = torch.device("cuda" if config.get("use_gpu", True) and torch.cuda.is_available() else "cpu")
            
            # 5. 初始化特征转换层
            self.feature_transforms = nn.ModuleDict({
                modality_type.value: nn.Sequential(
                    nn.Linear(dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ).to(self.device)
                for modality_type, dim in self.feature_dims.items()
            })
            
            # 6. 初始化注意力机制
            self.attention = nn.MultiheadAttention(
                embed_dim=512,
                num_heads=8,
                dropout=0.1
            ).to(self.device)
            
            # 7. 初始化融合层
            self.fusion_layers = nn.Sequential(
                nn.Linear(512 * len(ModalityType), 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 512)
            ).to(self.device)
            
            # 8. 初始化统计信息
            self.stats = {
                "fusion_times": deque(maxlen=100),
                "modality_counts": {m: 0 for m in ModalityType},
                "confidence_history": deque(maxlen=100)
            }
            
            logger.info("多模态融合系统初始化完成")
            
        except Exception as e:
            logger.error(f"多模态融合系统初始化失败: {e}")
            raise
        
    def add_feature(self, feature: ModalityFeature):
        """添加特征"""
        try:
            # 1. 基本验证
            if not isinstance(feature, ModalityFeature):
                raise ValueError("无效的特征类型")
                
            if feature.confidence < self.min_confidence:
                logger.warning(f"特征置信度过低: {feature.confidence}")
                return
                
            # 2. 检查特征维度
            expected_dim = self.feature_dims.get(feature.type)
            if expected_dim is None:
                logger.warning(f"未知的模态类型: {feature.type}")
                return
                
            if feature.feature.shape[0] != expected_dim:
                logger.warning(f"特征维度不匹配: {feature.type} 期望 {expected_dim}, 实际 {feature.feature.shape[0]}")
                return
                
            # 3. 添加到缓冲区
            self.feature_buffer[feature.type].append(feature)
            
            # 4. 更新统计信息
            self.stats["modality_counts"][feature.type] += 1
            
            # 5. 清理过期特征
            self._clean_expired_features()
            
        except Exception as e:
            logger.error(f"添加特征失败: {e}")
            
    def fuse(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """执行多模态特征融合"""
        try:
            # 1. 获取最新特征
            current_features = {}
            current_time = time.time()
            
            for modality_type in ModalityType:
                features = self.feature_buffer.get(modality_type, [])
                if not features:
                    continue
                    
                # 获取最新的特征
                latest_feature = max(features, key=lambda x: x.timestamp)
                
                # 检查时间窗口
                if current_time - latest_feature.timestamp > self.time_window:
                    continue
                    
                current_features[modality_type] = latest_feature.feature
                
            # 2. 检查是否有足够的特征
            if len(current_features) < self.anomaly_thresholds["min_modalities"]:
                return None, {"error": "特征数量不足"}
                
            # 3. 检查特征一致性
            if not self._check_feature_consistency(current_features):
                return None, {"error": "特征一致性检查失败"}
                
            # 4. 特征融合
            fused_feature = self._fuse_features(current_features)
            if fused_feature is None:
                return None, {"error": "特征融合失败"}
                
            # 5. 返回结果
            return fused_feature, {
                "num_features": len(current_features),
                "modalities": [m.value for m in current_features.keys()],
                "confidence": self._calculate_confidence(current_features)
            }
            
        except Exception as e:
            logger.error(f"特征融合失败: {e}")
            return None, {"error": str(e)}
            
    def _clean_expired_features(self):
        """清理过期特征"""
        try:
            current_time = time.time()
            for modality_type in ModalityType:
                self.feature_buffer[modality_type] = [
                    f for f in self.feature_buffer[modality_type]
                    if current_time - f.timestamp <= self.time_window
                ]
        except Exception as e:
            logger.error(f"清理过期特征失败: {e}")
            
    def _fuse_features(self, features: Dict[ModalityType, np.ndarray]) -> Optional[np.ndarray]:
        """特征融合"""
        try:
            # 1. 特征转换
            transformed_features = {}
            for modality_type, feature in features.items():
                tensor = torch.from_numpy(feature).float().to(self.device)
                transformed = self.feature_transforms[modality_type.value](tensor)
                transformed_features[modality_type] = transformed
                
            # 2. 堆叠特征用于注意力机制
            stacked_features = torch.stack(list(transformed_features.values()), dim=0)
            
            # 3. 应用注意力机制
            attended_features, _ = self.attention(
                stacked_features,
                stacked_features,
                stacked_features
            )
            
            # 4. 展平并连接特征
            flat_features = torch.cat([
                feat.mean(dim=0) for feat in attended_features.split(1, dim=0)
            ], dim=0)
            
            # 5. 最终融合
            fused_feature = self.fusion_layers(flat_features)
            
            return fused_feature.detach().cpu().numpy()
            
        except Exception as e:
            logger.error(f"特征融合失败: {e}")
            return None
            
    def _calculate_confidence(self, features: Dict[ModalityType, np.ndarray]) -> float:
        """计算融合置信度"""
        try:
            if not features:
                return 0.0
                
            # 1. 时间一致性分数
            timestamps = [f.timestamp for f in features.values()]
            time_diff = max(timestamps) - min(timestamps)
            time_score = max(0, 1 - time_diff / self.anomaly_thresholds["time_diff"])
            
            # 2. 特征一致性分数
            feature_scores = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    sim = 1 - cosine(features[i], features[j])
                    feature_scores.append(sim)
            feature_score = np.mean(feature_scores) if feature_scores else 0
            
            # 3. 模态完整性分数
            modality_score = len(features) / len(ModalityType)
            
            # 4. 各模态置信度
            confidence_score = np.mean([f.confidence for f in features.values()])
            
            # 综合评分
            weights = [0.3, 0.3, 0.2, 0.2]  # 各项权重
            final_score = np.average(
                [time_score, feature_score, modality_score, confidence_score],
                weights=weights
            )
            
            return final_score
            
        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return 0.0
            
    def _check_feature_consistency(self, features: Dict[ModalityType, np.ndarray]) -> bool:
        """检查特征一致性"""
        try:
            # 1. 检查特征维度
            dimensions = self.feature_dims
            
            for modality_type, feature in features.items():
                expected_dim = dimensions.get(modality_type)
                if expected_dim is None:
                    logger.warning(f"未知的模态类型: {modality_type}")
                    return False
                    
                if feature.shape[0] != expected_dim:
                    logger.warning(f"特征维度不匹配: {modality_type} 期望 {expected_dim}, 实际 {feature.shape[0]}")
                    # 进行维度调整
                    if feature.shape[0] < expected_dim:
                        # 填充
                        feature = np.pad(feature, (0, expected_dim - feature.shape[0]), 'constant')
                    else:
                        # 截断
                        feature = feature[:expected_dim]
                    features[modality_type] = feature
                    
            # 2. 检查特征值范围
            for feature in features.values():
                if np.any(np.isnan(feature)) or np.any(np.isinf(feature)):
                    logger.warning("特征包含无效值")
                    return False
                    
            # 3. 检查特征相似度
            for type1, feature1 in features.items():
                for type2, feature2 in features.items():
                    if type1 != type2:
                        similarity = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
                        if similarity > self.similarity_threshold:
                            logger.warning(f"特征相似度过高: {type1} 和 {type2}")
                            return False
                            
            return True
            
        except Exception as e:
            logger.error(f"特征一致性检查失败: {e}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        try:
            return {
                "avg_fusion_time": np.mean(list(self.stats["fusion_times"])) if self.stats["fusion_times"] else 0,
                "modality_counts": dict(self.stats["modality_counts"]),
                "avg_confidence": np.mean(list(self.stats["confidence_history"])) if self.stats["confidence_history"] else 0
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {} 