#!/usr/bin/env python3
"""直接测试transformers模型加载"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from loguru import logger

def test_direct_model():
    """直接测试模型加载和推理"""
    try:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        
        logger.info(f"开始加载模型: {model_name}")
        
        # 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        logger.info("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        logger.info("✅ 模型加载成功！")
        
        # 简单测试推理
        logger.info("测试推理...")
        prompt = "你好"
        
        # 使用chat模板
        messages = [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt")
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        logger.info(f"✅ 推理成功！")
        logger.info(f"用户: {prompt}")
        logger.info(f"助手: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("直接测试Qwen模型")
    logger.info("=" * 50)
    
    success = test_direct_model()
    
    if success:
        logger.info("🎉 测试完全成功！")
    else:
        logger.error("❌ 测试失败") 