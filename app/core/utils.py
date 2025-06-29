import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os

def compute_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """计算两个向量的余弦相似度，自动转为float64"""
    if feat1 is None or feat2 is None:
        return 0.0
    try:
        feat1 = np.array(feat1, dtype=np.float64)
        feat2 = np.array(feat2, dtype=np.float64)
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        return float(similarity)
    except Exception:
        return 0.0

def get_default_aes_key():
    # 32字节密钥（256位AES），实际部署请安全管理
    return os.environ.get('PROFILE_AES_KEY', 'default_key_32bytes_123456789012345').encode()[:32]

def encrypt_data(data: bytes, key: bytes = None) -> bytes:
    if key is None:
        key = get_default_aes_key()
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(data, AES.block_size))
    return cipher.iv + ct_bytes

def decrypt_data(data: bytes, key: bytes = None) -> bytes:
    if key is None:
        key = get_default_aes_key()
    iv = data[:16]
    ct = data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size) 