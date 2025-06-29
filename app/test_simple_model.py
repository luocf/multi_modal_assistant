#!/usr/bin/env python3
"""简单的模型加载测试"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from models.model_factory import ModelFactory
from config.model_config import ModelType

def test_simple_loading():
    """测试简单的模型加载"""
    try:
        logger.info("开始测试简单模型加载...")
        
        # 测试轻量级模型（默认）
        logger.info("正在加载轻量级模型...")
        model = ModelFactory.create_model(ModelType.LIGHT.value)
        
        # 获取模型信息
        model_info = model.get_model_info()
        logger.info(f"模型信息: {model_info}")
        
        # 检查模型类型
        if hasattr(model, 'model'):
            logger.info("✅ 真正的模型加载成功！")
            logger.info(f"模型类型: {type(model.model)}")
            return True
        else:
            logger.warning("❌ 仍在使用虚拟模型")
            return False
            
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False

def main():
    logger.info("=" * 50)
    logger.info("开始简单模型加载测试")
    logger.info("=" * 50)
    
    success = test_simple_loading()
    
    if success:
        logger.info("✅ 测试成功！真正的模型已加载")
    else:
        logger.error("❌ 测试失败！请检查模型配置")

if __name__ == "__main__":
    main() 