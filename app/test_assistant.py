from core.architecture import MultiModalAssistant
from loguru import logger

def test_assistant():
    """测试助手功能"""
    try:
        # 创建助手实例
        logger.info("正在初始化助手...")
        assistant = MultiModalAssistant()
        
        # 启动助手
        logger.info("正在启动助手...")
        assistant.start()
        
        # 测试对话功能
        test_prompts = [
            "你好，请介绍一下你自己",
            "你能做什么？",
            "今天天气怎么样？"
        ]
        
        for prompt in test_prompts:
            logger.info(f"\n用户: {prompt}")
            response = assistant.process_input(text=prompt)
            logger.info(f"助手: {response}")
            
        # 获取系统状态
        state = assistant.get_system_state()
        logger.info(f"\n系统状态: {state}")
        
        # 停止助手
        logger.info("\n正在停止助手...")
        assistant.stop()
        
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    test_assistant() 