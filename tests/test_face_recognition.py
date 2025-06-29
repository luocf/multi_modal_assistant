import pytest
import numpy as np
import cv2
from pathlib import Path
from core.perception.face_recognition import FaceRecognizer

@pytest.fixture
def face_recognizer():
    """创建人脸识别器实例"""
    storage_path = Path("tests/data/faces")
    storage_path.mkdir(parents=True, exist_ok=True)
    return FaceRecognizer(str(storage_path))

@pytest.fixture
def sample_image():
    """创建测试图像"""
    # 创建一个简单的测试图像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # 添加一个简单的矩形作为"人脸"
    cv2.rectangle(image, (200, 100), (400, 300), (255, 255, 255), -1)
    return image

def test_face_detection(face_recognizer, sample_image):
    """测试人脸检测功能"""
    face_locations = face_recognizer.detect_faces(sample_image)
    assert isinstance(face_locations, list)
    
def test_face_recognition(face_recognizer, sample_image):
    """测试人脸识别功能"""
    # 添加测试人脸
    face_encoding = np.random.rand(128)  # 模拟人脸特征向量
    user_id = "test_user"
    face_recognizer.add_face(face_encoding, user_id)
    
    # 测试识别
    recognized_user = face_recognizer.recognize_face(face_encoding)
    assert recognized_user == user_id
    
def test_process_frame(face_recognizer, sample_image):
    """测试图像帧处理功能"""
    results = face_recognizer.process_frame(sample_image)
    assert isinstance(results, list)
    
def test_invalid_image(face_recognizer):
    """测试无效图像处理"""
    invalid_image = np.zeros((10, 10, 3), dtype=np.uint8)
    results = face_recognizer.process_frame(invalid_image)
    assert len(results) == 0 