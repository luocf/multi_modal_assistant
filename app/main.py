import cv2
import numpy as np
import pyaudio
import wave
import threading
import time
from typing import Optional, Dict, Any
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from app.core.architecture import (
    MultiModalAssistant,
    SystemConfig,
    SystemState
)

class AudioCapture:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = []
        self.silence_frames = 0
        self.max_silence_frames = int(config.get("silence_limit", 1.0) * config["rate"] / config["chunk"])
        
    def start(self):
        """启动音频采集"""
        if self.stream is None:
            self.stream = self.audio.open(
                format=self.config["format"],  # 直接使用配置中的格式
                channels=self.config["channels"],
                rate=self.config["rate"],
                input=True,
                frames_per_buffer=self.config["chunk"]
            )
        self.is_recording = True
        self.frames = []
        self.silence_frames = 0
        
    def stop(self):
        """停止音频采集"""
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"关闭音频流失败: {e}")
            finally:
                self.stream = None
            
    def get_audio_data(self) -> np.ndarray:
        """获取音频数据"""
        if not self.is_recording or self.stream is None:
            return np.array([])
            
        try:
            data = self.stream.read(self.config["chunk"], exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # 只返回音频数据，不进行任何自动停止逻辑
            return audio_data
            
        except Exception as e:
            logger.error(f"读取音频数据失败: {e}")
            return np.array([])
            
    def __del__(self):
        """清理资源"""
        self.stop()
        if self.audio:
            self.audio.terminate()

class VideoCapture:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cap = None
        
    def start(self):
        """启动视频采集"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
            
    def stop(self):
        """停止视频采集"""
        if self.cap:
            self.cap.release()
            
    def get_frame(self) -> Optional[np.ndarray]:
        """获取视频帧"""
        if not self.cap:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        return frame

def draw_text_cn(img, text, pos, color=(255,255,255), font_size=24):
    # img: OpenCV格式BGR
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    # 初始化系统配置
    config = SystemConfig()
    
    # 修复音频配置格式
    audio_config = {
        "format": pyaudio.paInt16,  # 使用PyAudio格式常量
        "channels": config.audio_config["channels"],
        "rate": config.audio_config["rate"],
        "chunk": config.audio_config["chunk"],
        "threshold": 100,  # 添加音量阈值
        "silence_limit": 1.0  # 添加静音限制
    }
    
    # 创建助手实例
    assistant = MultiModalAssistant()
    
    # 启动系统
    assistant.start()
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        return
        
    # 初始化音频采集
    audio_capture = AudioCapture(audio_config)  # 使用修复后的配置
    audio_capture.start()  # 启动音频采集
    
    logger.info("系统已启动，按 'q' 退出")
    logger.info("开始说话，系统会自动检测语音并识别")
    logger.info("按 't' 进入文本输入模式")
    
    audio_frames = []
    last_recognition_time = time.time()
    min_recognition_interval = 2.0  # 增加最小识别间隔到2秒
    text_input_mode = False
    current_text = ""
    
    # 音频缓冲区管理
    audio_buffer = []
    min_audio_length = 16000 * 2  # 至少2秒的音频数据
    max_audio_length = 16000 * 10  # 最多10秒的音频数据
    silence_threshold = 50  # 提高静音阈值
    consecutive_silence_frames = 0
    max_silence_frames = 15  # 减少静音帧数阈值
    
    # 添加音频质量检测
    total_volume_sum = 0
    frame_count = 0
    
    # 新增：图像推理限频参数
    last_image_infer_time = 0
    image_infer_interval = 0.5  # 每0.5秒推理一次
    
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            logger.error("无法读取视频帧")
            break
            
        # 处理输入
        try:
            # 将视频帧转换为字节
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            
            response = None  # 初始化response变量
            
            # 获取音频数据
            if audio_capture.is_recording and not text_input_mode:
                logger.debug(f"音频采集状态: is_recording={audio_capture.is_recording}, text_input_mode={text_input_mode}")
                audio_data = audio_capture.get_audio_data()
                logger.debug(f"获取到音频数据长度: {len(audio_data)}")
                if len(audio_data) > 0:
                    # 添加到缓冲区
                    audio_buffer.extend(audio_data)
                    frame_count += 1
                    
                    # 计算音量
                    volume = np.abs(audio_data).mean()
                    max_volume = np.abs(audio_data).max()
                    total_volume_sum += volume
                    
                    # 检测静音
                    if volume < silence_threshold:
                        consecutive_silence_frames += 1
                    else:
                        consecutive_silence_frames = 0
                    
                    # 每20帧显示一次调试信息
                    if frame_count % 20 == 0:
                        avg_volume_so_far = total_volume_sum / frame_count
                        logger.debug(f"缓冲区长度: {len(audio_buffer)}, 当前音量: {volume:.2f}, 平均音量: {avg_volume_so_far:.2f}, 连续静音帧: {consecutive_silence_frames}")
                    
                    # 检查是否应该进行识别
                    should_recognize = False
                    
                    # 条件1：达到最小长度且检测到足够的静音
                    if len(audio_buffer) >= min_audio_length and consecutive_silence_frames >= max_silence_frames:
                        should_recognize = True
                        logger.info("检测到语音结束（静音）")
                    
                    # 条件2：达到最大长度
                    elif len(audio_buffer) >= max_audio_length:
                        should_recognize = True
                        logger.info("达到最大录音长度")
                    
                    # 条件3：简化的强制识别（每4秒且有足够音量）
                    elif len(audio_buffer) >= 16000 * 4:  # 4秒
                        avg_volume_so_far = total_volume_sum / frame_count
                        if avg_volume_so_far > 20:  # 平均音量足够
                            should_recognize = True
                            logger.info(f"强制识别（4秒，平均音量: {avg_volume_so_far:.2f}）")
                    
                    # 进行语音识别
                    if should_recognize:
                        current_time = time.time()
                        if current_time - last_recognition_time >= min_recognition_interval:
                            try:
                                # 将列表转换为numpy数组
                                audio_data_combined = np.array(audio_buffer, dtype=np.int16)
                                audio_bytes = audio_data_combined.tobytes()
                                
                                # 添加音频统计信息
                                total_length = len(audio_data_combined)
                                duration = total_length / 16000
                                avg_volume = np.abs(audio_data_combined).mean()
                                max_vol = np.abs(audio_data_combined).max()
                                
                                logger.info(f"准备识别音频: 长度={total_length}, 时长={duration:.2f}秒, 平均音量={avg_volume:.2f}, 最大音量={max_vol:.2f}")
                                
                                # 只有音量足够大才进行识别
                                if avg_volume > 10:
                                    # 保存音频文件用于调试
                                    try:
                                        with wave.open("debug_audio.wav", "wb") as wav_file:
                                            wav_file.setnchannels(1)
                                            wav_file.setsampwidth(2)
                                            wav_file.setframerate(16000)
                                            wav_file.writeframes(audio_bytes)
                                        logger.info("音频已保存到 debug_audio.wav")
                                    except Exception as e:
                                        logger.warning(f"保存音频文件失败: {e}")
                                    
                                    # 使用流式输出处理音频输入
                                    response_text = ""
                                    for chunk in assistant.process_input_stream(audio=audio_bytes, image=image_bytes):
                                        if chunk:
                                            response_text += chunk
                                            # 实时显示在视频帧上
                                            frame_with_text = frame.copy()
                                            frame_with_text = draw_text_cn(frame_with_text, f"Response: {response_text}", (10, 30))
                                            cv2.imshow("Multi-modal Assistant", frame_with_text)
                                            cv2.waitKey(1)
                                    response = response_text
                                    if response:
                                        logger.info(f"语音识别结果: {response}")
                                else:
                                    logger.warning(f"音量过低({avg_volume:.2f})，跳过识别")
                                
                            except Exception as e:
                                logger.error(f"音频数据处理失败: {e}")
                            
                            # 清空缓冲区和统计
                            audio_buffer = []
                            consecutive_silence_frames = 0
                            total_volume_sum = 0
                            frame_count = 0
                            last_recognition_time = current_time
                else:
                    logger.debug("音频数据为空")
            else:
                logger.debug(f"跳过音频处理: is_recording={audio_capture.is_recording}, text_input_mode={text_input_mode}")
            
            # 只每隔0.5秒推理一次，降低CPU占用
            current_time = time.time()
            if response is None and (current_time - last_image_infer_time > image_infer_interval):
                response = assistant.process_input(image=image_bytes)
                last_image_infer_time = current_time
            
            # 显示结果
            if response:
                frame = draw_text_cn(frame, f"Response: {response}", (10, 30))
                
            # 显示当前输入模式
            mode_text = "文本输入模式" if text_input_mode else "语音输入模式"
            frame = draw_text_cn(frame, f"Mode: {mode_text}", (10, 60))
            
            # 在文本输入模式下显示当前文本
            if text_input_mode:
                frame = draw_text_cn(frame, f"Text: {current_text}", (10, 90))
                
        except Exception as e:
            logger.error(f"处理输入失败: {e}")
            
        # 显示图像
        cv2.imshow("Multi-modal Assistant", frame)
        
        # 检查按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            text_input_mode = not text_input_mode
            if text_input_mode:
                # 停止音频采集
                audio_capture.stop()
                current_text = ""
                logger.info("进入文本输入模式")
            else:
                # 重新开始音频采集
                audio_capture.start()
                logger.info("进入语音输入模式")
        elif text_input_mode:
            # 处理文本输入
            if key == 13:  # Enter键
                if current_text:
                    response = assistant.process_input(text=current_text)
                    if response:
                        logger.info(f"文本输入结果: {response}")
                    current_text = ""
            elif key == 8:  # Backspace键
                current_text = current_text[:-1]
            elif 32 <= key <= 126:  # 可打印字符
                current_text += chr(key)
        time.sleep(0.02)  # 限制主循环帧率，降低CPU占用
        
    # 清理资源
    audio_capture.stop()
    cap.release()
    cv2.destroyAllWindows()
    assistant.stop()

if __name__ == "__main__":
    main() 