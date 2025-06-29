from typing import Dict, Any
import psutil
from app.config.model_config import ModelConfig, ModelType
from loguru import logger

class MemoryManager:
    def __init__(self):
        self.allocated_memory = 0
        
    def allocate_memory(self, model_name: str, required_memory: int) -> bool:
        """分配内存"""
        available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        logger.info(f"系统可用内存: {available_memory:.2f}GB")
        logger.info(f"模型所需内存: {required_memory}GB")
        
        # 预留1GB系统内存
        if available_memory - 1 >= required_memory:
            self.allocated_memory += required_memory
            return True
        return False
        
    def release_memory(self, memory: int):
        """释放内存"""
        self.allocated_memory -= memory

class ModelLoader:
    def __init__(self):
        self.loaded_models = {}
        self.memory_manager = MemoryManager()
        self.model_config = ModelConfig()
        
    def load_model(self, model_type: str):
        """按需加载模型"""
        try:
            if model_type in self.loaded_models:
                return self.loaded_models[model_type]
                
            config = self.model_config.get_model_config(model_type)
            if not self.memory_manager.allocate_memory(model_type, config["max_memory"]):
                # 如果内存不足，尝试加载更小的模型
                if model_type == ModelType.MEDIUM.value:
                    logger.warning("内存不足，尝试加载轻量级模型...")
                    return self.load_model(ModelType.LIGHT.value)
                elif model_type == ModelType.HEAVY.value:
                    logger.warning("内存不足，尝试加载中等模型...")
                    return self.load_model(ModelType.MEDIUM.value)
                else:
                    raise RuntimeError(f"内存不足，无法加载模型 {config['name']}")
                    
            model = self._load_model_with_config(model_type, config)
            self.loaded_models[model_type] = model
            return model
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
            
    def _release_unused_models(self):
        """释放未使用的模型"""
        for model_type, model in list(self.loaded_models.items()):
            if not self._is_model_in_use(model):
                config = self.model_config.get_model_config(model_type)
                self.memory_manager.release_memory(config["max_memory"])
                del self.loaded_models[model_type]
                
    def _is_model_in_use(self, model) -> bool:
        """检查模型是否在使用中"""
        # TODO: 实现模型使用状态检查
        return False
        
    def _load_model_with_config(self, model_type: str, config: dict):
        """根据配置加载模型"""
        try:
            if config.get("type") == "qwen_cloud":
                from .qwen_cloud_model import QwenCloudModel
                logger.info(f"正在加载云端Qwen模型: {config['name']}")
                return QwenCloudModel(config)
            # 其它模型类型仍然返回DummyModel（如需恢复本地模型可再调整）
            logger.warning(f"已禁用本地Qwen模型，使用虚拟模型: {config['name']}")
            return DummyModel(config)
        except Exception as e:
            logger.error(f"加载模型配置失败: {e}，使用虚拟模型作为后备")
            return DummyModel(config)

class DummyModel:
    """临时使用的虚拟模型（仅作为后备）"""
    def __init__(self, config: dict):
        self.config = config
        logger.warning("使用虚拟模型，这只是一个占位符实现")
        
    def generate(self, prompt: str) -> str:
        try:
            if not prompt or not isinstance(prompt, str):
                return "抱歉，输入无效。"
            logger.warning("虚拟模型被调用，请检查真正的模型是否正确加载")
            return f"[虚拟模型回复] 抱歉，真正的AI模型暂时不可用。您的输入是：{prompt[:100]}..."
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return "抱歉，生成回复时出现错误。"
        
    def get_model_info(self) -> Dict[str, Any]:
        return self.config

class ModelFactory:
    _instance = None
    _loader = None
    
    @classmethod
    def create_model(cls, model_type: str = None):
        """创建模型实例"""
        try:
            if cls._instance is None:
                cls._instance = cls()
                cls._loader = ModelLoader()
                
            if model_type is None:
                model_type = ModelConfig().get_default_model()
                
            return cls._loader.load_model(model_type)
        except Exception as e:
            logger.error(f"创建模型实例失败: {e}")
            raise 