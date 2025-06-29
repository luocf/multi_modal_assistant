from typing import Dict, Any
from enum import Enum
import os

class ModelType(Enum):
    """支持的模型类型"""
    LIGHT = "light"      # 轻量级模型，适合资源受限环境
    MEDIUM = "medium"    # 中等规模模型，平衡性能和资源消耗
    HEAVY = "heavy"      # 大规模模型，需要更多资源
    CLOUD = "cloud"       # 云端模型

class ModelConfig:
    """模型配置"""
    def __init__(self):
        self.configs = {
            ModelType.CLOUD.value: {
                "name": "Qwen/Qwen2.5-1.5B-Cloud",
                "type": "qwen_cloud",
                "description": "云端Qwen2.5模型，无本地资源限制",
                "max_memory": 0,  # 云端模型不占用本地内存
                "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
                "scenarios": ["通用对话", "语音交互", "图像理解"],
                "system_requirements": {
                    "device": "cloud",
                    "memory": "0GB"
                },
                "use_case": "对话"
            },
            ModelType.LIGHT.value: {
                "name": "Qwen/Qwen2.5-1.5B-Instruct",  # 使用最新的2.5版本
                "quantization": "int8",  # 使用int8量化
                "device": "cpu",
                "max_memory": 2,  # GB，降低内存要求
                "description": "轻量级模型，适合资源受限环境",
                "use_case": "对话"
            },
            ModelType.MEDIUM.value: {
                "name": "Qwen/Qwen2.5-7B-Instruct",  # 使用最新的2.5版本
                "quantization": "int4",  # 使用int4量化
                "device": "cpu",
                "max_memory": 12,  # GB，提高内存要求
                "description": "中等规模模型，平衡性能和资源消耗",
                "use_case": "对话"
            },
            ModelType.HEAVY.value: {
                "name": "Qwen/Qwen2.5-72B-Instruct",  # 使用最新的2.5版本
                "quantization": "int4",  # 使用int4量化
                "device": "cuda",  # 需要GPU
                "max_memory": 24,  # GB，提高内存要求
                "description": "大规模模型，需要更多资源",
                "use_case": "对话"
            }
        }
        
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取指定类型的模型配置"""
        if model_type not in self.configs:
            raise ValueError(f"未知的模型类型: {model_type}")
        return self.configs[model_type]
        
    def get_default_model(self) -> str:
        """获取默认模型类型"""
        return ModelType.CLOUD.value  # 默认使用云端模型
        
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """获取模型信息"""
        config = self.get_model_config(model_type)
        # 优先从config['device']获取，其次从system_requirements.device，最后默认'cpu'
        device = config.get("device")
        if not device:
            device = config.get("system_requirements", {}).get("device", "cpu")
        return {
            "name": config["name"],
            "description": config["description"],
            "use_case": config["use_case"],
            "requirements": {
                "device": device,
                "memory": f"{config['max_memory']}GB"
            }
        }

qwen_cloud_config = {
    "name": "Qwen/Qwen2.5-1.5B-Cloud",
    "type": "qwen_cloud",
    "model": "qwen-plus",  # 可根据实际云端模型名调整
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    # "api_key": "xxx",  # 通过环境变量获取
} 