@echo off

:: 创建虚拟环境
python -m venv venv
call venv\Scripts\activate.bat

:: 安装基础依赖
pip install -r requirements.txt

:: 安装PyTorch (CPU版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

:: 下载insightface模型
python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l', root='./models')"

echo 安装完成！
echo 请使用 'venv\Scripts\activate.bat' 激活虚拟环境 