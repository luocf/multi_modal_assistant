#!/usr/bin/env python3
"""
混合架构配置脚本
- 云端对话：使用阿里云Qwen模型
- 本地感知：人脸识别、声纹识别、表情识别、手势识别
"""

import os
import subprocess
import sys
from pathlib import Path

def install_dashscope():
    """安装dashscope SDK"""
    print("🔧 正在安装阿里云dashscope SDK...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "dashscope"])
        print("✅ dashscope SDK安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ dashscope SDK安装失败: {e}")
        return False

def setup_env_file():
    """设置环境变量文件"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("📝 创建.env文件...")
        with open(env_file, "w", encoding="utf-8") as f:
            f.write("# 阿里云DashScope API密钥\n")
            f.write("# 请在 https://dashscope.console.aliyun.com/ 获取API密钥\n")
            f.write("DASHSCOPE_API_KEY=your_api_key_here\n")
            f.write("\n# 其他配置\n")
            f.write("MODEL_TYPE=cloud\n")
        print("✅ .env文件创建成功")
    else:
        print("ℹ️  .env文件已存在")
    
    # 检查API密钥是否配置
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("\n⚠️  请配置DASHSCOPE_API_KEY:")
        print("1. 访问 https://dashscope.console.aliyun.com/")
        print("2. 注册/登录阿里云账号")
        print("3. 获取API密钥")
        print("4. 在.env文件中设置 DASHSCOPE_API_KEY=你的密钥")
        return False
    
    return True

def verify_local_models():
    """验证本地模型文件"""
    models_dir = Path("models")
    
    print("🔍 检查本地模型文件...")
    
    # 检查人脸识别模型
    face_model_dir = models_dir / "models" / "buffalo_l"
    if face_model_dir.exists():
        print("✅ 人脸识别模型 (InsightFace) 已就绪")
    else:
        print("⚠️  人脸识别模型未找到，首次运行时会自动下载")
    
    # 检查声纹识别模型
    voice_model_dir = models_dir / "spkrec-ecapa-voxceleb"
    if voice_model_dir.exists():
        print("✅ 声纹识别模型 (SpeechBrain) 已就绪")
    else:
        print("⚠️  声纹识别模型未找到，首次运行时会自动下载")
    
    return True

def update_model_config():
    """更新模型配置，确保使用云端对话模型"""
    config_file = Path("app/config/model_config.py")
    
    print("📝 更新模型配置...")
    
    # 读取当前配置
    with open(config_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 确保默认使用云端模型
    if 'return ModelType.CLOUD.value' not in content:
        content = content.replace(
            'return ModelType.LIGHT.value',
            'return ModelType.CLOUD.value'
        )
        
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        print("✅ 模型配置已更新为云端模式")
    else:
        print("ℹ️  模型配置已是云端模式")

def create_run_script():
    """创建启动脚本"""
    script_content = '''#!/bin/bash
# 混合架构多模态语音助手启动脚本

echo "🚀 启动混合架构多模态语音助手"
echo "📡 对话模型：阿里云Qwen (云端)"
echo "👁️  感知功能：本地处理 (人脸/声纹/表情/手势)"
echo ""

# 检查环境变量
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "⚠️  未设置DASHSCOPE_API_KEY环境变量"
    echo "请在.env文件中配置您的API密钥"
    exit 1
fi

# 启动应用
python -m app.main
'''
    
    with open("run_hybrid.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod("run_hybrid.sh", 0o755)
    print("✅ 启动脚本创建成功: run_hybrid.sh")

def main():
    """主函数"""
    print("🎯 配置混合架构多模态语音助手")
    print("=" * 50)
    
    # 1. 安装依赖
    if not install_dashscope():
        return False
    
    # 2. 设置环境变量
    if not setup_env_file():
        print("\n❌ 请先配置API密钥再继续")
        return False
    
    # 3. 验证本地模型
    verify_local_models()
    
    # 4. 更新配置
    update_model_config()
    
    # 5. 创建启动脚本
    create_run_script()
    
    print("\n🎉 混合架构配置完成！")
    print("\n📋 架构说明:")
    print("  💬 对话生成: 阿里云Qwen模型 (云端)")
    print("  👁️  人脸识别: InsightFace (本地)")
    print("  🎤 声纹识别: SpeechBrain (本地)")
    print("  😊 表情识别: FER (本地)")
    print("  👋 手势识别: MediaPipe (本地)")
    
    print("\n🚀 启动方式:")
    print("  方式1: ./run_hybrid.sh")
    print("  方式2: python -m app.main")
    
    print("\n💡 注意事项:")
    print("  1. 确保网络连接正常 (云端对话需要)")
    print("  2. 首次运行会下载本地模型文件")
    print("  3. 云端API有调用频率限制")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 