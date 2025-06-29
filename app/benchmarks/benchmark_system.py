import time
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import psutil
import torch
from loguru import logger
from tqdm import tqdm

from core.enhancement.audio_enhancer import AudioEnhancer
from core.enhancement.video_enhancer import VideoEnhancer
from core.identity.multimodal_fusion import MultiModalFusion, ModalityFeature, ModalityType
from core.monitoring.performance_monitor import PerformanceMonitor, Metric, MetricType
from config.enhancement_config import (
    AUDIO_ENHANCER_CONFIG,
    VIDEO_ENHANCER_CONFIG,
    MULTIMODAL_FUSION_CONFIG,
    PERFORMANCE_MONITOR_CONFIG
)

class SystemBenchmark:
    def __init__(self):
        """初始化基准测试"""
        self.audio_enhancer = AudioEnhancer(AUDIO_ENHANCER_CONFIG)
        self.video_enhancer = VideoEnhancer(VIDEO_ENHANCER_CONFIG)
        self.fusion_system = MultiModalFusion(MULTIMODAL_FUSION_CONFIG)
        self.performance_monitor = PerformanceMonitor(PERFORMANCE_MONITOR_CONFIG)
        
        # 测试配置
        self.test_duration = 60  # 测试持续时间（秒）
        self.batch_sizes = [1, 2, 4, 8, 16]  # 测试不同批量大小
        self.num_threads = [1, 2, 4, 8]  # 测试不同线程数
        
        logger.info("基准测试初始化完成")
        
    def generate_test_data(self, num_samples: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """生成测试数据"""
        audio_samples = []
        video_samples = []
        
        for _ in range(num_samples):
            # 生成音频数据
            duration = 1.0
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440 * t)
            audio += np.random.normal(0, 0.1, len(audio))
            audio_samples.append(audio)
            
            # 生成视频帧
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.ellipse(image, (320, 240), (100, 130), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(image, (280, 200), 20, (0, 0, 0), -1)
            cv2.circle(image, (360, 200), 20, (0, 0, 0), -1)
            image += np.random.normal(0, 25, image.shape).astype(np.uint8)
            video_samples.append(image)
            
        return audio_samples, video_samples
        
    def run_single_test(self, audio: np.ndarray, video: np.ndarray) -> Dict[str, float]:
        """运行单个测试"""
        try:
            start_time = time.time()
            
            # 1. 音频增强
            audio_start = time.time()
            enhanced_audio, _ = self.audio_enhancer.enhance(audio)
            audio_time = time.time() - audio_start
            
            # 2. 视频增强
            video_start = time.time()
            enhanced_video, _ = self.video_enhancer.enhance(video)
            video_time = time.time() - video_start
            
            # 3. 特征提取（模拟）
            feature_start = time.time()
            features = {
                ModalityType.FACE: np.random.rand(512),
                ModalityType.VOICE: np.random.rand(256),
                ModalityType.LIP: np.random.rand(128),
                ModalityType.BEHAVIOR: np.random.rand(64)
            }
            feature_time = time.time() - feature_start
            
            # 4. 特征融合
            fusion_start = time.time()
            current_time = time.time()
            for modality_type, feature in features.items():
                self.fusion_system.add_feature(ModalityFeature(
                    type=modality_type,
                    feature=feature,
                    confidence=0.8,
                    timestamp=current_time,
                    metadata={}
                ))
            fused_feature, _ = self.fusion_system.fuse()
            fusion_time = time.time() - fusion_start
            
            total_time = time.time() - start_time
            
            return {
                "audio_time": audio_time,
                "video_time": video_time,
                "feature_time": feature_time,
                "fusion_time": fusion_time,
                "total_time": total_time
            }
            
        except Exception as e:
            logger.error(f"测试执行失败: {e}")
            return {}
            
    def run_batch_test(self, batch_size: int, num_threads: int) -> Dict[str, Any]:
        """运行批量测试"""
        try:
            logger.info(f"运行批量测试: batch_size={batch_size}, threads={num_threads}")
            
            # 生成测试数据
            audio_samples, video_samples = self.generate_test_data(batch_size)
            
            # 使用线程池执行测试
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for audio, video in zip(audio_samples, video_samples):
                    futures.append(
                        executor.submit(self.run_single_test, audio, video)
                    )
                
                # 收集结果
                results = []
                for future in futures:
                    results.append(future.result())
                    
            # 计算统计信息
            stats = {
                "audio_time": [],
                "video_time": [],
                "feature_time": [],
                "fusion_time": [],
                "total_time": []
            }
            
            for result in results:
                for key in stats:
                    if key in result:
                        stats[key].append(result[key])
                        
            # 计算平均值和标准差
            summary = {}
            for key, values in stats.items():
                if values:
                    summary[f"{key}_mean"] = np.mean(values)
                    summary[f"{key}_std"] = np.std(values)
                    
            # 添加系统资源使用情况
            system_stats = self.performance_monitor.get_system_stats()
            summary.update({
                "cpu_usage": system_stats.get("cpu_percent", 0),
                "memory_usage": system_stats.get("memory_percent", 0)
            })
            
            if "gpu_memory_allocated" in system_stats:
                summary["gpu_memory"] = system_stats["gpu_memory_allocated"]
                
            return summary
            
        except Exception as e:
            logger.error(f"批量测试失败: {e}")
            return {}
            
    def run_benchmark(self):
        """运行完整基准测试"""
        try:
            logger.info("开始运行基准测试...")
            results = []
            
            # 测试不同配置
            for batch_size in self.batch_sizes:
                for num_threads in self.num_threads:
                    # 运行测试
                    result = self.run_batch_test(batch_size, num_threads)
                    result.update({
                        "batch_size": batch_size,
                        "num_threads": num_threads
                    })
                    results.append(result)
                    
                    # 记录性能指标
                    self.performance_monitor.add_metric(Metric(
                        type=MetricType.THROUGHPUT,
                        name="processing_speed",
                        value=batch_size / result.get("total_time_mean", 1),
                        timestamp=time.time(),
                        metadata=result
                    ))
                    
            # 生成报告
            self._generate_report(results)
            
        except Exception as e:
            logger.error(f"基准测试失败: {e}")
            
    def _generate_report(self, results: List[Dict[str, Any]]):
        """生成测试报告"""
        try:
            logger.info("\n=== 基准测试报告 ===")
            
            # 1. 系统信息
            logger.info("\n系统信息:")
            logger.info(f"CPU: {psutil.cpu_count()} cores")
            logger.info(f"内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                
            # 2. 性能摘要
            logger.info("\n性能摘要:")
            for result in results:
                logger.info(
                    f"\nBatch Size: {result['batch_size']}, "
                    f"Threads: {result['num_threads']}"
                )
                logger.info(
                    f"平均处理时间: {result.get('total_time_mean', 0):.3f}秒 "
                    f"(±{result.get('total_time_std', 0):.3f})"
                )
                logger.info(
                    f"处理速度: {result['batch_size']/result.get('total_time_mean', 1):.1f} "
                    f"样本/秒"
                )
                logger.info(f"CPU使用率: {result.get('cpu_usage', 0):.1f}%")
                logger.info(f"内存使用率: {result.get('memory_usage', 0):.1f}%")
                
            # 3. 各组件性能分析
            logger.info("\n组件性能分析:")
            for result in results:
                if result['batch_size'] == max(self.batch_sizes):
                    logger.info(f"\n最大批量({result['batch_size']})时的组件耗时:")
                    logger.info(
                        f"音频增强: {result.get('audio_time_mean', 0):.3f}秒 "
                        f"(±{result.get('audio_time_std', 0):.3f})"
                    )
                    logger.info(
                        f"视频增强: {result.get('video_time_mean', 0):.3f}秒 "
                        f"(±{result.get('video_time_std', 0):.3f})"
                    )
                    logger.info(
                        f"特征提取: {result.get('feature_time_mean', 0):.3f}秒 "
                        f"(±{result.get('feature_time_std', 0):.3f})"
                    )
                    logger.info(
                        f"特征融合: {result.get('fusion_time_mean', 0):.3f}秒 "
                        f"(±{result.get('fusion_time_std', 0):.3f})"
                    )
                    
            # 4. 建议
            logger.info("\n优化建议:")
            # 找出瓶颈
            bottleneck = max(results[-1], key=lambda x: results[-1].get(f"{x}_time_mean", 0))
            logger.info(f"- 主要瓶颈在于{bottleneck}阶段，建议优先优化")
            
            # 分析线程扩展性
            thread_scaling = []
            for num_threads in self.num_threads:
                for result in results:
                    if result['num_threads'] == num_threads and result['batch_size'] == max(self.batch_sizes):
                        thread_scaling.append(result.get('total_time_mean', 0))
            if len(thread_scaling) > 1:
                scaling_efficiency = thread_scaling[0] / thread_scaling[-1] / len(thread_scaling)
                if scaling_efficiency < 0.7:
                    logger.info("- 线程扩展性不佳，建议优化并行处理逻辑")
                    
            # 分析内存使用
            max_memory = max(result.get('memory_usage', 0) for result in results)
            if max_memory > 80:
                logger.info("- 内存使用率过高，建议优化内存管理")
                
            # 分析GPU使用
            if any('gpu_memory' in result for result in results):
                gpu_util = max(result.get('gpu_memory', 0) for result in results)
                if gpu_util < 0.5:
                    logger.info("- GPU利用率较低，建议优化GPU计算任务")
                    
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            
if __name__ == '__main__':
    benchmark = SystemBenchmark()
    benchmark.run_benchmark() 