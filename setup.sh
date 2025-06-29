#!/bin/bash

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"

# 创建虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装基础依赖
echo "安装基础依赖..."
pip install -r requirements.txt

# 安装PyTorch (CPU版本)
echo "安装PyTorch..."
pip install torch torchvision torchaudio

# 创建模型目录
echo "创建模型目录..."
mkdir -p models

# 下载insightface模型
echo "下载insightface模型..."
python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l', root='./models')"

echo "安装完成！"
echo "请使用 'source venv/bin/activate' 激活虚拟环境"
echo "然后运行 'python app/main.py' 启动程序" 