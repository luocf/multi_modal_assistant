#!/bin/bash
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
