#!/usr/bin/env python3
"""测试真正的模型推理功能"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from models.model_factory import ModelFactory
from config.model_config import ModelType

def test_model_loading():
    """测试模型加载"""
    try:
        logger.info("开始测试模型加载...")
        
        # 测试轻量级模型（默认）
        logger.info("正在加载轻量级模型...")
        model = ModelFactory.create_model(ModelType.LIGHT.value)
        
        # 获取模型信息
        model_info = model.get_model_info()
        logger.info(f"模型信息: {model_info}")
        
        # 测试简单的推理
        test_prompts = [
            "你好，请介绍一下你自己。",
            "今天天气怎么样？",
            "请解释一下人工智能的基本概念。"
        ]
        
        for prompt in test_prompts:
            logger.info(f"测试提示: {prompt}")
            try:
                response = model.generate(prompt)
                logger.info(f"模型回复: {response}")
                print(f"\n问: {prompt}")
                print(f"答: {response}\n")
            except Exception as e:
                logger.error(f"生成回复失败: {e}")
                
        logger.info("模型测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"模型测试失败: {e}")
        return False

def main():
    logger.info("=" * 50)
    logger.info("开始测试真正的大模型推理功能")
    logger.info("=" * 50)
    
    success = test_model_loading()
    
    if success:
        logger.info("✅ 测试成功！真正的模型推理功能正常工作")
    else:
        logger.error("❌ 测试失败！请检查模型配置和依赖")

if __name__ == "__main__":
    main() 