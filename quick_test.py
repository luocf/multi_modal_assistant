#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置API密钥
api_key = os.getenv("DASHSCOPE_API_KEY")
print(f"API密钥: {api_key[:8]}..." if api_key else "未设置API密钥")

try:
    import dashscope
    from dashscope import Generation
    
    # 设置API密钥
    dashscope.api_key = api_key
    
    print("正在测试API调用...")
    
    # 测试API调用
    response = Generation.call(
        model='qwen-plus',
        messages=[
            {'role': 'user', 'content': '你好'}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"响应对象: {type(response)}")
    print(f"响应内容: {response}")
    
    # 尝试解析响应
    if hasattr(response, 'status_code'):
        print(f"状态码: {response.status_code}")
    
    if hasattr(response, 'output'):
        print(f"输出: {response.output}")
        
        if hasattr(response.output, 'choices'):
            print(f"选择: {response.output.choices}")
            if response.output.choices:
                content = response.output.choices[0]['message']['content']
                print(f"✅ 成功获取回复: {content}")
        elif hasattr(response.output, 'text'):
            print(f"✅ 成功获取回复: {response.output.text}")
            
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc() 