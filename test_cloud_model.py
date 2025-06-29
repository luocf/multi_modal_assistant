#!/usr/bin/env python3
"""
测试云端模型配置和API密钥
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def test_env_loading():
    """测试环境变量加载"""
    print("🔍 检查环境变量...")
    
    # 尝试加载.env文件
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env文件加载成功")
    except ImportError:
        print("⚠️  python-dotenv未安装，尝试手动加载环境变量")
        # 手动读取.env文件
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value
            print("✅ 手动加载.env文件成功")
    
    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        print(f"✅ API密钥已设置: {api_key[:8]}...")
        return True
    else:
        print("❌ API密钥未设置")
        return False

def test_dashscope_import():
    """测试dashscope模块导入"""
    print("\n🔍 检查dashscope模块...")
    try:
        import dashscope
        print("✅ dashscope模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ dashscope模块导入失败: {e}")
        print("请运行: pip install dashscope")
        return False

def test_model_config():
    """测试模型配置"""
    print("\n🔍 检查模型配置...")
    try:
        from app.config.model_config import ModelConfig
        config = ModelConfig()
        cloud_config = config.get_model_config("cloud")
        print(f"✅ 云端模型配置: {cloud_config['name']}")
        
        api_key = cloud_config.get("api_key")
        if api_key:
            print(f"✅ 配置中的API密钥: {api_key[:8]}...")
            return True
        else:
            print("❌ 配置中未找到API密钥")
            return False
    except Exception as e:
        print(f"❌ 模型配置加载失败: {e}")
        return False

def test_cloud_model():
    """测试云端模型"""
    print("\n🔍 测试云端模型...")
    try:
        from app.models.qwen_cloud_model import QwenCloudModel
        from app.config.model_config import ModelConfig
        
        config = ModelConfig()
        cloud_config = config.get_model_config("cloud")
        
        model = QwenCloudModel(cloud_config)
        print("✅ 云端模型初始化成功")
        
        # 测试简单对话
        print("🤖 测试对话...")
        response = model.generate("你好")
        print(f"✅ 对话测试成功: {response[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ 云端模型测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 云端模型配置测试")
    print("=" * 50)
    
    success = True
    
    # 1. 测试环境变量
    if not test_env_loading():
        success = False
    
    # 2. 测试dashscope导入
    if not test_dashscope_import():
        success = False
    
    # 3. 测试模型配置
    if not test_model_config():
        success = False
    
    # 4. 测试云端模型
    if success and not test_cloud_model():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 所有测试通过！云端模型配置正常")
        print("\n💡 现在可以运行主程序:")
        print("   python -m app.main")
    else:
        print("❌ 部分测试失败，请检查配置")
        
        print("\n🔧 修复建议:")
        print("1. 确保已安装dashscope: pip install dashscope")
        print("2. 检查.env文件中的API密钥配置")
        print("3. 确保网络连接正常")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 