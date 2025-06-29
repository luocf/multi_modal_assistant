from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from loguru import logger
from app.models.model_factory import ModelFactory
from app.config.model_config import ModelType, ModelConfig

class DialogueManager:
    def __init__(self, model_type: str = None):
        """
        初始化对话管理器
        
        Args:
            model_type: 模型类型，默认使用 qwen
        """
        try:
            self.model_config = ModelConfig()
            if model_type is None:
                model_type = self.model_config.get_default_model()
                
            # 获取模型信息
            model_info = self.model_config.get_model_info(model_type)
            logger.info(f"使用模型: {model_info['name']}")
            logger.info(f"模型描述: {model_info['description']}")
            logger.info(f"适用场景: {model_info['use_case']}")
            logger.info(f"系统要求: {model_info['requirements']}")
            
            # 创建模型实例
            self.model = ModelFactory.create_model(model_type)
            
            # 设置系统提示词
            self.system_prompt = """你是一个多模态AI助手，具有以下能力：
1. 语音识别和理解
2. 图像识别和理解
3. 自然语言对话
4. 情感分析
5. 手势识别

请根据用户的输入（可能是语音、图像或文本）提供合适的回复。
回复应该：
1. 简洁明了
2. 符合上下文
3. 考虑用户的情感状态
4. 适当使用表情符号
5. 保持友好和专业的语气

如果遇到无法理解的内容，请诚实地表示，并尝试引导用户提供更清晰的信息。"""
            
            logger.info("对话管理器初始化成功")
            
            # 初始化对话历史
            self.dialogue_history = []
            self.max_history_length = 10
            
        except Exception as e:
            logger.error(f"初始化对话管理器失败: {e}")
            raise
            
    def _format_prompt(self, user_input: str, user_profile: Optional[Dict] = None) -> str:
        """
        格式化提示词
        
        Args:
            user_input: 用户输入
            user_profile: 用户画像数据
            
        Returns:
            格式化后的提示词
        """
        try:
            # 基础提示词
            prompt = "你是一个智能助手，请根据用户输入提供帮助。\n\n"
            
            # 添加用户画像信息
            if user_profile:
                prompt += f"用户信息：\n"
                if user_profile.get("tags"):
                    prompt += f"用户标签：{', '.join(user_profile['tags'])}\n"
                if user_profile.get("emotion_history"):
                    recent_emotions = user_profile["emotion_history"][-3:]
                    prompt += f"最近情绪：{', '.join([e['emotion'] for e in recent_emotions])}\n"
                    
            # 添加对话历史
            if self.dialogue_history:
                prompt += "\n对话历史：\n"
                for turn in self.dialogue_history[-self.max_history_length:]:
                    prompt += f"用户：{turn['user']}\n"
                    prompt += f"助手：{turn['assistant']}\n"
                    
            # 添加当前输入
            prompt += f"\n用户：{user_input}\n助手："
            
            return prompt
            
        except Exception as e:
            logger.error(f"格式化提示词失败: {e}")
            return user_input
            
    def _update_history(self, user_input: str, assistant_response: str):
        """
        更新对话历史
        
        Args:
            user_input: 用户输入
            assistant_response: 助手回复
        """
        try:
            self.dialogue_history.append({
                "user": user_input,
                "assistant": assistant_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # 保持历史记录在限制范围内
            if len(self.dialogue_history) > self.max_history_length:
                self.dialogue_history = self.dialogue_history[-self.max_history_length:]
                
        except Exception as e:
            logger.error(f"更新对话历史失败: {e}")
            
    def get_response(self, user_input: str, user_profile: Optional[Dict] = None) -> str:
        """
        获取助手回复
        
        Args:
            user_input: 用户输入
            user_profile: 用户画像数据
            
        Returns:
            助手回复
        """
        try:
            # 格式化提示词
            prompt = self._format_prompt(user_input, user_profile)
            
            # 生成回复
            response = self.model.generate(prompt)
            
            # 更新历史记录
            self._update_history(user_input, response)
            
            return response
            
        except Exception as e:
            logger.error(f"获取助手回复失败: {e}")
            return "抱歉，我现在无法回应，请稍后再试。"
            
    def clear_history(self):
        """清空对话历史"""
        self.dialogue_history = []
        
    def get_history(self) -> List[Dict]:
        """
        获取对话历史
        
        Returns:
            对话历史列表
        """
        return self.dialogue_history
        
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return self.model.get_model_info()

    def generate_response(self, prompt: str = None, audio_text: str = None, image_description: str = None, user_profile: Dict[str, Any] = None, user_id: str = None) -> str:
        """生成回复"""
        try:
            if not prompt and not audio_text and not image_description:
                return "抱歉，我没有收到任何输入。"
                
            # 构建完整的提示词
            full_prompt = self.system_prompt + "\n\n"
            
            # 添加用户信息
            if user_profile and user_id:
                full_prompt += f"用户信息:\n"
                full_prompt += f"- 用户ID: {user_id}\n"
                full_prompt += f"- 姓名: {user_profile.get('name', '未知')}\n"
                full_prompt += f"- 访问次数: {user_profile.get('visit_count', 0)}\n"
                full_prompt += f"- 用户标签: {', '.join(user_profile.get('tags', []))}\n"
                full_prompt += f"- 上次见面: {user_profile.get('last_seen', '未知')}\n"
                if user_profile.get('is_frequent_visitor'):
                    full_prompt += "- 这是一位常客，请表现得更加亲切熟悉\n"
                full_prompt += "\n"
            elif user_id:
                if user_id == 'visitor':
                    full_prompt += "用户信息: 这是一位访客，请保持礼貌和专业\n\n"
                else:
                    full_prompt += f"用户信息: 已识别用户 {user_id}\n\n"
            
            if audio_text:
                full_prompt += f"用户语音输入: {audio_text}\n"
            if image_description:
                full_prompt += f"图像内容: {image_description}\n"
            if prompt:
                full_prompt += f"用户文本输入: {prompt}\n"
                
            full_prompt += "\n请根据以上信息提供合适的回复。如果是熟悉的用户，可以更加亲切自然。"
            
            response = self.model.generate(full_prompt)
            if not response:
                return "抱歉，我无法生成回复。"
                
            return response
            
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return "抱歉，生成回复时出现错误。"

    def generate_response_stream(self, prompt: str = None, audio_text: str = None, image_description: str = None, user_profile: Dict[str, Any] = None, user_id: str = None):
        """流式生成回复，返回生成器"""
        try:
            if not prompt and not audio_text and not image_description:
                yield "抱歉，我没有收到任何输入。"
                return
                
            # 构建完整的提示词
            full_prompt = self.system_prompt + "\n\n"
            
            # 添加用户信息
            if user_profile and user_id:
                full_prompt += f"用户信息:\n"
                full_prompt += f"- 用户ID: {user_id}\n"
                full_prompt += f"- 姓名: {user_profile.get('name', '未知')}\n"
                full_prompt += f"- 访问次数: {user_profile.get('visit_count', 0)}\n"
                full_prompt += f"- 用户标签: {', '.join(user_profile.get('tags', []))}\n"
                full_prompt += f"- 上次见面: {user_profile.get('last_seen', '未知')}\n"
                if user_profile.get('is_frequent_visitor'):
                    full_prompt += "- 这是一位常客，请表现得更加亲切熟悉\n"
                full_prompt += "\n"
            elif user_id:
                if user_id == 'visitor':
                    full_prompt += "用户信息: 这是一位访客，请保持礼貌和专业\n\n"
                else:
                    full_prompt += f"用户信息: 已识别用户 {user_id}\n\n"
            
            if audio_text:
                full_prompt += f"用户语音输入: {audio_text}\n"
            if image_description:
                full_prompt += f"图像内容: {image_description}\n"
            if prompt:
                full_prompt += f"用户文本输入: {prompt}\n"
                
            full_prompt += "\n请根据以上信息提供合适的回复。如果是熟悉的用户，可以更加亲切自然。"
            
            # 判断模型是否支持流式
            if hasattr(self.model, "generate_stream"):
                for chunk in self.model.generate_stream(full_prompt):
                    yield chunk
            else:
                # 不支持流式则整体返回
                yield self.model.generate(full_prompt)
        except Exception as e:
            logger.error(f"流式生成回复失败: {e}")
            yield "抱歉，流式生成回复时出现错误。" 