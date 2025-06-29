from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel
import soundfile as sf
from loguru import logger

from core.enhancement.audio_enhancer import AudioEnhancer
from core.enhancement.video_enhancer import VideoEnhancer
from core.identity.multimodal_fusion import MultiModalFusion, ModalityFeature, ModalityType
from core.monitoring.performance_monitor import PerformanceMonitor, Metric, MetricType
from config.enhancement_config import (
    AUDIO_ENHANCER_CONFIG,
    VIDEO_ENHANCER_CONFIG,
    MULTIMODAL_FUSION_CONFIG,
    PERFORMANCE_MONITOR_CONFIG
)

# 创建FastAPI应用
app = FastAPI(
    title="多模态身份识别系统",
    description="提供音频增强、视频增强和多模态融合服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化组件
audio_enhancer = AudioEnhancer(AUDIO_ENHANCER_CONFIG)
video_enhancer = VideoEnhancer(VIDEO_ENHANCER_CONFIG)
fusion_system = MultiModalFusion(MULTIMODAL_FUSION_CONFIG)
performance_monitor = PerformanceMonitor(PERFORMANCE_MONITOR_CONFIG)

# 定义请求/响应模型
class EnhancementResponse(BaseModel):
    success: bool
    message: str
    stats: Dict[str, Any]
    enhanced_data: Optional[str] = None

class FusionRequest(BaseModel):
    features: Dict[str, list]
    timestamp: float
    metadata: Dict[str, Any] = {}

class FusionResponse(BaseModel):
    success: bool
    message: str
    fused_feature: Optional[list] = None
    confidence: float
    stats: Dict[str, Any]

@app.post("/enhance/audio", response_model=EnhancementResponse)
async def enhance_audio(audio_file: UploadFile = File(...)):
    """
    增强音频质量
    - 支持格式：WAV, FLAC, OGG
    - 最大文件大小：10MB
    """
    try:
        # 1. 读取音频文件
        content = await audio_file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="文件大小超过限制")
            
        # 2. 解码音频
        audio_data, sample_rate = sf.read(io.BytesIO(content))
        if sample_rate != AUDIO_ENHANCER_CONFIG["sample_rate"]:
            raise HTTPException(status_code=400, detail="采样率不支持")
            
        # 3. 增强音频
        start_time = time.time()
        enhanced_audio, stats = audio_enhancer.enhance(audio_data)
        
        # 4. 编码增强后的音频
        output = io.BytesIO()
        sf.write(output, enhanced_audio, sample_rate, format='WAV')
        enhanced_bytes = output.getvalue()
        
        # 5. 记录性能指标
        processing_time = time.time() - start_time
        performance_monitor.add_metric(Metric(
            type=MetricType.LATENCY,
            name="audio_processing_time",
            value=processing_time,
            timestamp=time.time(),
            metadata=stats
        ))
        
        return EnhancementResponse(
            success=True,
            message="音频增强成功",
            stats=stats,
            enhanced_data=enhanced_bytes.hex()  # 转换为十六进制字符串
        )
        
    except Exception as e:
        logger.error(f"音频增强失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/enhance/video", response_model=EnhancementResponse)
async def enhance_video(video_file: UploadFile = File(...)):
    """
    增强视频帧质量
    - 支持格式：JPEG, PNG
    - 最大文件大小：5MB
    """
    try:
        # 1. 读取图像文件
        content = await video_file.read()
        if len(content) > 5 * 1024 * 1024:  # 5MB
            raise HTTPException(status_code=400, detail="文件大小超过限制")
            
        # 2. 解码图像
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="图像解码失败")
            
        # 3. 增强图像
        start_time = time.time()
        enhanced_image, stats = video_enhancer.enhance(image)
        
        # 4. 编码增强后的图像
        _, enhanced_bytes = cv2.imencode('.jpg', enhanced_image)
        
        # 5. 记录性能指标
        processing_time = time.time() - start_time
        performance_monitor.add_metric(Metric(
            type=MetricType.LATENCY,
            name="video_processing_time",
            value=processing_time,
            timestamp=time.time(),
            metadata=stats
        ))
        
        return EnhancementResponse(
            success=True,
            message="视频帧增强成功",
            stats=stats,
            enhanced_data=enhanced_bytes.tobytes().hex()
        )
        
    except Exception as e:
        logger.error(f"视频增强失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/fusion", response_model=FusionResponse)
async def fuse_features(request: FusionRequest):
    """
    执行多模态特征融合
    - 需要提供各模态的特征向量
    - 特征维度必须符合配置要求
    """
    try:
        # 1. 验证特征维度
        for modality, feature in request.features.items():
            expected_dim = MULTIMODAL_FUSION_CONFIG["feature_dims"].get(modality)
            if expected_dim != len(feature):
                raise HTTPException(
                    status_code=400,
                    detail=f"特征维度不匹配: {modality} 期望{expected_dim}维，实际{len(feature)}维"
                )
                
        # 2. 添加特征
        start_time = time.time()
        for modality, feature in request.features.items():
            modality_feature = ModalityFeature(
                type=ModalityType[modality.upper()],
                feature=np.array(feature),
                confidence=0.8,  # 可以从请求中获取
                timestamp=request.timestamp,
                metadata=request.metadata
            )
            fusion_system.add_feature(modality_feature)
            
        # 3. 执行融合
        fused_feature, stats = fusion_system.fuse()
        
        # 4. 记录性能指标
        processing_time = time.time() - start_time
        performance_monitor.add_metric(Metric(
            type=MetricType.LATENCY,
            name="fusion_time",
            value=processing_time,
            timestamp=time.time(),
            metadata=stats
        ))
        
        return FusionResponse(
            success=True,
            message="特征融合成功",
            fused_feature=fused_feature.tolist() if fused_feature is not None else None,
            confidence=stats.get("confidence", 0.0),
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"特征融合失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/stats")
async def get_system_stats():
    """获取系统性能统计"""
    try:
        # 获取各类统计信息
        metrics = performance_monitor.get_metrics()
        system_stats = performance_monitor.get_system_stats()
        
        return JSONResponse({
            "success": True,
            "metrics": metrics,
            "system_stats": system_stats
        })
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/reset")
async def reset_system():
    """重置系统状态"""
    try:
        # 重置各组件
        audio_enhancer.reset_stats()
        video_enhancer.reset_stats()
        fusion_system.reset_stats()
        performance_monitor.reset_stats()
        
        return JSONResponse({
            "success": True,
            "message": "系统状态已重置"
        })
        
    except Exception as e:
        logger.error(f"重置系统失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 