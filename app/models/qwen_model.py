from typing import Dict, Any, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseModel
from loguru import logger

class QwenModel(BaseModel):
    """Qwen 模型实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Qwen 模型
        
        Args:
            config: 配置字典，包含模型名称、设备和最大内存
        """
        super().__init__(config)
        self.model_name = config["name"]
        self.device = config["device"]
        self.max_memory = config["max_memory"]
        self._load_model()
        
    def _load_model(self):
        """加载模型"""
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 根据设备选择合适的数据类型
            if self.device == "cuda" and torch.cuda.is_available():
                torch_dtype = torch.float16  # GPU使用float16
                device_map = "auto"
            else:
                torch_dtype = torch.float32  # CPU使用float32
                device_map = "cpu"
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )
            
            # 设置生成参数
            self.generation_config = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "do_sample": True
            }
            
            logger.info(f"模型加载成功: {self.model_name}")
            logger.info(f"使用设备: {device_map}")
            logger.info(f"数据类型: {torch_dtype}")
            
        except Exception as e:
            logger.error(f"模型加载错误: {str(e)}")
            raise
            
    def generate(self, prompt: str) -> str:
        """生成回复"""
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
                
            # 解码输出
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # 提取回复部分
            response = response[len(prompt):].strip()
            return response
            
        except Exception as e:
            logger.error(f"生成回复错误: {str(e)}")
            return "抱歉，我现在无法生成回复。"
            
    def get_memory_usage(self) -> float:
        """获取模型内存使用量（GB）"""
        if hasattr(self.model, "get_memory_footprint"):
            return self.model.get_memory_footprint() / (1024 ** 3)
        return 0.0

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "name": self.model_name,
            "type": "qwen",
            "size": "1.5B",
            "device": str(self.device)
        } 