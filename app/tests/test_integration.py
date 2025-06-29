import unittest
import numpy as np
import cv2
import time
from typing import Dict, Any
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.core.enhancement.audio_enhancer import AudioEnhancer
from app.core.enhancement.video_enhancer import VideoEnhancer
from app.core.identity.multimodal_fusion import MultiModalFusion, ModalityFeature, ModalityType
from app.core.monitoring.performance_monitor import PerformanceMonitor, Metric, MetricType
from app.config.enhancement_config import (
    AUDIO_ENHANCER_CONFIG,
    VIDEO_ENHANCER_CONFIG,
    MULTIMODAL_FUSION_CONFIG,
    PERFORMANCE_MONITOR_CONFIG
)

class TestSystemIntegration(unittest.TestCase):
    def setUp(self):
        """测试初始化"""
        # 初始化各个组件
        self.audio_enhancer = AudioEnhancer(AUDIO_ENHANCER_CONFIG)
        self.video_enhancer = VideoEnhancer(VIDEO_ENHANCER_CONFIG)
        self.fusion_system = MultiModalFusion(MULTIMODAL_FUSION_CONFIG)
        self.performance_monitor = PerformanceMonitor(PERFORMANCE_MONITOR_CONFIG)
        
        # 生成测试数据
        self._generate_test_data()
        
    def _generate_test_data(self):
        """生成测试数据"""
        # 1. 音频数据
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        self.test_audio = np.sin(2 * np.pi * 440 * t)
        self.noisy_audio = self.test_audio + np.random.normal(0, 0.1, len(self.test_audio))
        
        # 2. 视频数据
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 添加人脸形状
        cv2.ellipse(self.test_image, (320, 240), (100, 130), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(self.test_image, (280, 200), 20, (0, 0, 0), -1)  # 左眼
        cv2.circle(self.test_image, (360, 200), 20, (0, 0, 0), -1)  # 右眼
        self.noisy_image = self.test_image + np.random.normal(0, 25, self.test_image.shape).astype(np.uint8)
        
    def test_full_pipeline(self):
        """测试完整处理流程"""
        try:
            # 1. 音频增强
            enhanced_audio, audio_stats = self.audio_enhancer.enhance(self.noisy_audio)
            self.assertIsNotNone(enhanced_audio)
            self.performance_monitor.add_metric(Metric(
                type=MetricType.QUALITY,
                name="audio_quality",
                value=audio_stats.get("enhanced_volume", 0),
                timestamp=time.time(),
                metadata=audio_stats
            ))
            
            # 2. 视频增强
            enhanced_video, video_stats = self.video_enhancer.enhance(self.noisy_image)
            self.assertIsNotNone(enhanced_video)
            self.performance_monitor.add_metric(Metric(
                type=MetricType.QUALITY,
                name="video_quality",
                value=video_stats.get("quality_score", 0),
                timestamp=time.time(),
                metadata=video_stats
            ))
            
            # 3. 特征提取（模拟）
            features = {
                ModalityType.FACE: self._extract_face_feature(enhanced_video),
                ModalityType.VOICE: self._extract_voice_feature(enhanced_audio),
                ModalityType.LIP: self._extract_lip_feature(enhanced_video),
                ModalityType.BEHAVIOR: self._extract_behavior_feature(enhanced_video)
            }
            
            # 4. 添加特征到融合系统
            current_time = time.time()
            for modality_type, feature in features.items():
                modality_feature = ModalityFeature(
                    type=modality_type,
                    feature=feature,
                    confidence=0.8,
                    timestamp=current_time,
                    metadata={}
                )
                self.fusion_system.add_feature(modality_feature)
                
            # 5. 执行融合
            fused_feature, fusion_stats = self.fusion_system.fuse()
            self.assertIsNotNone(fused_feature)
            self.performance_monitor.add_metric(Metric(
                type=MetricType.QUALITY,
                name="fusion_quality",
                value=fusion_stats.get("confidence", 0),
                timestamp=time.time(),
                metadata=fusion_stats
            ))
            
            # 6. 检查性能监控
            system_stats = self.performance_monitor.get_system_stats()
            self.assertIsInstance(system_stats, dict)
            self.assertIn("cpu_percent", system_stats)
            self.assertIn("memory_percent", system_stats)
            
        except Exception as e:
            self.fail(f"集成测试失败: {e}")
            
    def test_error_handling(self):
        """测试错误处理"""
        # 1. 测试无效音频输入
        invalid_audio = np.zeros(1000)
        enhanced_audio, stats = self.audio_enhancer.enhance(invalid_audio)
        self.assertIsNotNone(enhanced_audio)  # 应该返回原始音频
        
        # 2. 测试无效视频输入
        invalid_video = np.zeros((100, 100))  # 错误的通道数
        enhanced_video, stats = self.video_enhancer.enhance(invalid_video)
        self.assertIsNone(enhanced_video)  # 应该返回None
        
        # 3. 测试融合系统的错误处理
        # 添加不完整的特征集
        current_time = time.time()
        incomplete_feature = ModalityFeature(
            type=ModalityType.FACE,
            feature=np.random.rand(512),
            confidence=0.8,
            timestamp=current_time,
            metadata={}
        )
        self.fusion_system.add_feature(incomplete_feature)
        
        # 尝试融合不完整的特征集
        fused_feature, fusion_stats = self.fusion_system.fuse()
        self.assertIsNone(fused_feature)  # 应该返回None
        self.assertIn("error", fusion_stats)
        
    def test_performance_monitoring(self):
        """测试性能监控"""
        # 1. 添加各类型指标
        current_time = time.time()
        test_metrics = [
            Metric(MetricType.LATENCY, "processing_time", 0.1, current_time, {}),
            Metric(MetricType.THROUGHPUT, "fps", 30.0, current_time, {}),
            Metric(MetricType.RESOURCE, "cpu_usage", 50.0, current_time, {}),
            Metric(MetricType.QUALITY, "recognition_accuracy", 0.95, current_time, {}),
            Metric(MetricType.ERROR, "error_rate", 0.01, current_time, {})
        ]
        
        for metric in test_metrics:
            self.performance_monitor.add_metric(metric)
            
        # 2. 检查指标统计
        metrics = self.performance_monitor.get_metrics()
        self.assertIsInstance(metrics, dict)
        for metric_type in MetricType:
            self.assertIn(metric_type.value, metrics)
            
        # 3. 检查系统资源监控
        system_stats = self.performance_monitor.get_system_stats()
        self.assertIsInstance(system_stats, dict)
        self.assertGreaterEqual(system_stats["cpu_percent"], 0)
        self.assertGreaterEqual(system_stats["memory_percent"], 0)
        
    def _extract_face_feature(self, image: np.ndarray) -> np.ndarray:
        """模拟人脸特征提取"""
        return np.random.rand(512)  # 模拟512维特征向量
        
    def _extract_voice_feature(self, audio: np.ndarray) -> np.ndarray:
        """模拟声纹特征提取"""
        return np.random.rand(512)  # 模拟512维特征向量
        
    def _extract_lip_feature(self, image: np.ndarray) -> np.ndarray:
        """模拟唇语特征提取"""
        return np.random.rand(512)  # 模拟512维特征向量
        
    def _extract_behavior_feature(self, image: np.ndarray) -> np.ndarray:
        """模拟行为特征提取"""
        return np.random.rand(512)  # 模拟512维特征向量
        
if __name__ == '__main__':
    unittest.main() 