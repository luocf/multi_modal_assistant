#!/bin/bash

# 检查conda环境
echo "当前conda环境: $CONDA_DEFAULT_ENV"

# 使用conda安装基础依赖
echo "安装基础依赖..."
conda install -y numpy opencv
conda install -y -c conda-forge insightface
conda install -y -c conda-forge transformers
conda install -y -c conda-forge onnxruntime
conda install -y pytorch torchvision torchaudio -c pytorch

# 创建模型目录
echo "创建模型目录..."
mkdir -p models

# 下载insightface模型
echo "下载insightface模型..."
python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l', root='./models')"

echo "安装完成！"
echo "请确保使用 'conda activate langgraphenv' 激活环境"
echo "然后运行 'python app/main.py' 启动程序" 