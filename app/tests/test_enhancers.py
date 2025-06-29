import unittest
import numpy as np
import torch
import cv2
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from app.core.enhancement import AudioEnhancer, VideoEnhancer
from app.config.enhancement_config import AUDIO_ENHANCER_CONFIG, VIDEO_ENHANCER_CONFIG

class TestAudioEnhancer(unittest.TestCase):
    def setUp(self):
        """测试初始化"""
        self.enhancer = AudioEnhancer(AUDIO_ENHANCER_CONFIG)
        
        # 生成测试音频数据
        self.sample_rate = 16000
        duration = 1.0  # 1秒
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        self.test_audio = np.sin(2 * np.pi * 440 * t)  # 440Hz音频
        self.noisy_audio = self.test_audio + np.random.normal(0, 0.1, len(self.test_audio))
        
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.enhancer.sample_rate, AUDIO_ENHANCER_CONFIG["sample_rate"])
        self.assertEqual(self.enhancer.frame_length, AUDIO_ENHANCER_CONFIG["frame_length"])
        
    def test_enhance(self):
        """测试音频增强"""
        enhanced, stats = self.enhancer.enhance(self.noisy_audio)
        
        # 检查输出
        self.assertIsNotNone(enhanced)
        self.assertEqual(len(enhanced), len(self.noisy_audio))
        self.assertIsInstance(stats, dict)
        
        # 检查信噪比是否提高
        original_snr = self._calculate_snr(self.test_audio, self.noisy_audio)
        enhanced_snr = self._calculate_snr(self.test_audio, enhanced)
        self.assertGreater(enhanced_snr, original_snr)
        
    def test_vad(self):
        """测试语音活动检测"""
        # 生成静音片段
        silence = np.zeros(self.sample_rate)
        is_speech = self.enhancer._is_speech(silence)
        self.assertFalse(is_speech)
        
        # 生成语音片段
        speech = self.test_audio
        is_speech = self.enhancer._is_speech(speech)
        self.assertTrue(is_speech)
        
    def _calculate_snr(self, clean: np.ndarray, noisy: np.ndarray) -> float:
        """计算信噪比"""
        noise = noisy - clean
        return 10 * np.log10(np.sum(clean**2) / np.sum(noise**2))
        
class TestVideoEnhancer(unittest.TestCase):
    def setUp(self):
        """测试初始化"""
        self.enhancer = VideoEnhancer(VIDEO_ENHANCER_CONFIG)
        
        # 生成测试图像
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(self.test_image, (320, 240), 100, (255, 255, 255), -1)
        
        # 添加噪声
        self.noisy_image = self.test_image.copy()
        self.noisy_image += np.random.normal(0, 25, self.test_image.shape).astype(np.uint8)
        
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.enhancer.target_fps, VIDEO_ENHANCER_CONFIG["target_fps"])
        self.assertEqual(self.enhancer.min_face_size, VIDEO_ENHANCER_CONFIG["min_face_size"])
        
    def test_enhance(self):
        """测试图像增强"""
        enhanced, stats = self.enhancer.enhance(self.noisy_image)
        
        # 检查输出
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, self.noisy_image.shape)
        self.assertIsInstance(stats, dict)
        
        # 检查图像质量是否提高
        original_quality = self.enhancer._calculate_quality(self.noisy_image)
        enhanced_quality = self.enhancer._calculate_quality(enhanced)
        self.assertGreater(enhanced_quality, original_quality)
        
    def test_face_detection(self):
        """测试人脸检测"""
        # 加载测试人脸图像
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 生成包含人脸的测试图像
        face_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.ellipse(face_img, (320, 240), (100, 130), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(face_img, (280, 200), 20, (0, 0, 0), -1)  # 左眼
        cv2.circle(face_img, (360, 200), 20, (0, 0, 0), -1)  # 右眼
        
        faces = self.enhancer._detect_faces(face_img)
        self.assertGreater(len(faces), 0)
        
    def test_quality_assessment(self):
        """测试图像质量评估"""
        # 测试清晰图像
        clear_quality = self.enhancer._calculate_quality(self.test_image)
        self.assertGreater(clear_quality, 0.7)
        
        # 测试模糊图像
        blurred = cv2.GaussianBlur(self.test_image, (21, 21), 0)
        blurred_quality = self.enhancer._calculate_quality(blurred)
        self.assertLess(blurred_quality, clear_quality)
        
if __name__ == '__main__':
    unittest.main() 