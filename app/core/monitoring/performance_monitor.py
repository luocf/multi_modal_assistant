import time
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np
from loguru import logger
import psutil
import threading
from dataclasses import dataclass
from enum import Enum
import GPUtil

class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE = "resource"
    QUALITY = "quality"
    ERROR = "error"

@dataclass
class Metric:
    type: MetricType
    name: str
    value: float
    timestamp: float
    metadata: Dict[str, Any]

class PerformanceMonitor:
    def __init__(self, config: Dict[str, Any]):
        """初始化性能监控器"""
        try:
            # 1. 初始化配置
            self.metrics_window = config.get("metrics_window", 60)
            self.log_interval = config.get("log_interval", 5)
            self.alert_thresholds = {
                "max_cpu": config.get("alert_thresholds", {}).get("cpu_percent", 80.0),
                "max_memory": config.get("alert_thresholds", {}).get("memory_percent", 80.0),
                "max_latency": config.get("alert_thresholds", {}).get("latency_ms", 100.0) / 1000.0,
                "min_fps": config.get("alert_thresholds", {}).get("min_fps", 15.0),
                "max_error_rate": config.get("alert_thresholds", {}).get("max_error_rate", 0.1),
                "min_quality": config.get("alert_thresholds", {}).get("min_quality", 0.5)
            }
            
            # 2. 初始化指标存储
            self.metrics: Dict[str, List[Metric]] = {}
            for metric_type in MetricType:
                self.metrics[metric_type.value] = []
            
            # 3. 初始化系统资源监控
            self.system_stats = {
                "start_time": time.time(),
                "gpu_stats": None
            }
            
            # 4. 检查GPU可用性
            if config.get("enable_gpu_monitoring", True):
                try:
                    gpu_stats = GPUtil.getGPUs()
                    if gpu_stats:
                        self.system_stats["gpu_stats"] = True
                        logger.info("检测到可用GPU")
                except Exception as e:
                    logger.warning(f"GPU检测失败: {e}")
            
            logger.info("性能监控器初始化完成")
            
        except Exception as e:
            logger.error(f"性能监控器初始化失败: {e}")
            raise
        
    def add_metric(self, metric: Metric):
        """添加新的性能指标"""
        try:
            # 更新指标存储
            self.metrics[metric.type.value].append(metric)
            
            # 检查是否需要记录日志
            current_time = time.time()
            if current_time - self.system_stats["start_time"] >= 1:
                self._log_metrics()
                self.system_stats["start_time"] = current_time
                
            # 检查告警条件
            self._check_alerts()
            
        except Exception as e:
            logger.error(f"添加指标失败: {e}")
            
    def get_metrics(self) -> Dict[str, List[Metric]]:
        """获取所有指标"""
        return self.metrics
            
    def _calculate_metrics_stats(self, metric_type: MetricType) -> Dict[str, Any]:
        """计算指标统计信息"""
        try:
            values = [m.value for m in self.metrics[metric_type.value]]
            if not values:
                return {}
                
            return {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
                "last_value": values[-1]
            }
        except Exception as e:
            logger.error(f"计算指标统计失败: {e}")
            return {}
            
    def _check_alerts(self) -> None:
        """检查是否需要触发告警"""
        try:
            # 1. 系统资源告警
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.alert_thresholds["max_cpu"]:
                self._add_alert("CPU使用率过高", cpu_percent)
                
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            if memory_percent > self.alert_thresholds["max_memory"]:
                self._add_alert("内存使用率过高", memory_percent)
                
            # 2. 性能指标告警
            metrics = self.get_metrics()
            
            # 检查延迟
            latency_metrics = metrics.get(MetricType.LATENCY.value, [])
            if latency_metrics:
                avg_latency = np.mean([m.value for m in latency_metrics])
                if avg_latency > self.alert_thresholds["max_latency"]:
                    self._add_alert("处理延迟过高", avg_latency)
                    
            # 检查吞吐量
            throughput_metrics = metrics.get(MetricType.THROUGHPUT.value, [])
            if throughput_metrics:
                avg_fps = np.mean([m.value for m in throughput_metrics])
                if avg_fps < self.alert_thresholds["min_fps"]:
                    self._add_alert("帧率过低", avg_fps)
                    
            # 检查错误率
            error_metrics = metrics.get(MetricType.ERROR.value, [])
            if error_metrics:
                error_rate = np.mean([m.value for m in error_metrics])
                if error_rate > self.alert_thresholds["max_error_rate"]:
                    self._add_alert("错误率过高", error_rate)
                    
            # 检查质量分数
            quality_metrics = metrics.get(MetricType.QUALITY.value, [])
            if quality_metrics:
                avg_quality = np.mean([m.value for m in quality_metrics])
                if avg_quality < self.alert_thresholds["min_quality"]:
                    self._add_alert("质量分数过低", avg_quality)
                    
        except Exception as e:
            logger.error(f"告警检查失败: {e}")
            
    def _log_metrics(self):
        """记录性能指标日志"""
        try:
            # 计算运行时间
            uptime = time.time() - self.system_stats["start_time"]
            
            # 获取系统资源使用情况
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            
            # 构建日志消息
            log_msg = (
                f"性能报告 (运行时间: {uptime:.1f}秒)\n"
                f"CPU使用率: {cpu_usage:.1f}%\n"
                f"内存使用率: {memory_usage:.1f}%"
            )
            
            # 添加GPU信息
            if self.system_stats["gpu_stats"]:
                gpu_percent = GPUtil.getGPUs()[0].load * 100
                log_msg += f"\nGPU使用率: {gpu_percent:.1f}%"
                
            # 添加各类型指标统计
            for metric_type in MetricType:
                stats = self._calculate_metrics_stats(metric_type)
                if stats:
                    log_msg += f"\n{metric_type.value}指标:"
                    log_msg += f" 均值={stats['mean']:.3f}"
                    log_msg += f" 标准差={stats['std']:.3f}"
                    
            logger.info(log_msg)
            
        except Exception as e:
            logger.error(f"记录指标日志失败: {e}")
            
    def reset_stats(self):
        """重置统计信息"""
        try:
            # 重置指标存储
            self.metrics = {}
            for metric_type in MetricType:
                self.metrics[metric_type.value] = []
                
            # 重置系统资源监控
            self.system_stats = {
                "start_time": time.time(),
                "gpu_stats": self.system_stats["gpu_stats"]
            }
            
            logger.info("性能统计已重置")
            
        except Exception as e:
            logger.error(f"重置统计信息失败: {e}")
            
    def get_system_stats(self) -> Dict[str, float]:
        """获取系统资源使用情况"""
        try:
            # 1. CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if np.isnan(cpu_percent):
                cpu_percent = 0.0
            
            # 2. 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            if np.isnan(memory_percent):
                memory_percent = 0.0
            
            # 3. GPU使用率（如果可用）
            gpu_percent = 0.0
            if self.system_stats["gpu_stats"]:
                try:
                    gpu_stats = GPUtil.getGPUs()
                    if gpu_stats:
                        gpu_percent = gpu_stats[0].load * 100
                except Exception as e:
                    logger.warning(f"获取GPU信息失败: {e}")
                
            return {
                "cpu_percent": float(cpu_percent),
                "memory_percent": float(memory_percent),
                "gpu_percent": float(gpu_percent)
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "gpu_percent": 0.0
            }
            
    def _add_alert(self, alert_type: str, value: float):
        """添加告警"""
        logger.warning(f"{alert_type}: {value:.2f}")
        
    def has_gpu(self):
        """检查是否存在GPU"""
        return self.system_stats["gpu_stats"] is not None 