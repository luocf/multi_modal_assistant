from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseModel(ABC):
    """基础模型接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化基础模型
        
        Args:
            config: 模型配置字典
        """
        self.config = config
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成回复
        
        Args:
            prompt: 输入提示词
            **kwargs: 其他参数
            
        Returns:
            生成的回复文本
        """
        pass
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        pass 