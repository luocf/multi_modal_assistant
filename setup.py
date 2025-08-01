from setuptools import setup, find_packages

setup(
    name="multi_model_demo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.7.0",
        "torchaudio>=0.7.0",
        "pyaudio>=0.2.11",
        "speechbrain>=0.5.12",
        "scipy>=1.7.0",
        "soundfile>=0.10.3",
        "librosa>=0.8.1",
        "deepface>=0.0.79",
        "opencv-contrib-python>=4.8.0",
        "mediapipe>=0.8.9",
        "onnxruntime>=1.8.0",
        "insightface>=0.2.1",
        "mtcnn>=1.0.0",
        "loguru>=0.7.0",
    ],
    python_requires=">=3.10",
    dependency_links=[
        "https://mirrors.aliyun.com/pypi/simple/",
        "https://pypi.tuna.tsinghua.edu.cn/simple/",
    ],
) 