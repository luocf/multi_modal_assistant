from typing import Dict, Any
import dashscope
from loguru import logger

class QwenCloudModel:
    def __init__(self, config: dict):
        self.config = config
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("未设置DASHSCOPE_API_KEY环境变量")
        dashscope.api_key = self.api_key
        
    def generate(self, prompt: str) -> str:
        """生成回复"""
        try:
            if not prompt or not isinstance(prompt, str):
                return "抱歉，输入无效。"
                
            # 使用新版本的DashScope API
            from dashscope import Generation
            
            response = Generation.call(
                model='qwen-plus',  # 使用qwen-plus模型
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                top_p=0.8,
                temperature=0.7,
                max_tokens=1500,
                stream=False
            )
            
            logger.debug(f"API响应: {response}")
            
            # 检查响应状态
            if hasattr(response, 'status_code') and response.status_code == 200:
                if hasattr(response, 'output') and response.output:
                    if hasattr(response.output, 'choices') and response.output.choices:
                        return response.output.choices[0]['message']['content']
                    elif hasattr(response.output, 'text'):
                        return response.output.text
                    else:
                        logger.error(f"API响应格式异常: {response.output}")
                        return "抱歉，API响应格式异常。"
                else:
                    logger.error("API响应中没有output字段")
                    return "抱歉，API响应格式异常。"
            else:
                error_msg = getattr(response, 'message', '未知错误')
                logger.error(f"API调用失败: {getattr(response, 'code', 'unknown')} - {error_msg}")
                return f"抱歉，生成回复失败: {error_msg}"
                
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return f"抱歉，生成回复时出现错误: {str(e)}"
            
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.config 