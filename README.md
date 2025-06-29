# 多模态智能语音助手 (Multi-Modal AI Assistant)

基于 Qwen 2.5 Omni 7B 的本地多模态语音助手，集成人脸识别、声纹识别、表情识别、手势识别等多种感知能力，支持无注册身份识别、动态画像补全和端到端语音对话。

## 🌟 核心特性

### 多模态感知能力
- **人脸识别**: 基于 InsightFace，支持实时人脸检测与识别
- **声纹识别**: 集成 SpeechBrain，实现声音身份验证
- **表情识别**: 使用 FER/DeepFace 进行情感状态分析
- **手势识别**: 基于 MediaPipe 的手势动作识别
- **唇语分析**: 支持唇语与语音的同步分析

### 智能身份管理
- **无注册识别**: 自动学习和记忆用户特征
- **多模态融合**: 综合人脸、声纹等多种特征进行身份确认
- **动态画像**: 实时更新用户偏好和行为模式
- **隐私保护**: 本地加密存储，支持数据脱敏

### 对话与交互
- **端到端语音对话**: 支持连续语音交互
- **个性化响应**: 基于用户画像提供定制化服务
- **多轮对话**: 维护对话上下文和会话状态
- **流式响应**: 实时生成和播放回复内容

### 系统特性
- **混合架构**: 本地感知 + 云端对话的最优配置
- **低延迟**: 优化的推理管道，响应时间 < 500ms
- **跨平台**: 支持 Windows、macOS、Linux
- **可扩展**: 模块化设计，易于功能扩展

## 🏗️ 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Application Layer)                 │
├─────────────────────────────────────────────────────────────┤
│  主控制器 (Main Controller)  │  API服务器 (API Server)      │
├─────────────────────────────────────────────────────────────┤
│                    核心层 (Core Layer)                       │
├─────────────────────────────────────────────────────────────┤
│  身份管理     │  对话管理     │  感知融合     │  用户画像     │
│ (Identity)   │ (Dialogue)   │ (Perception) │ (Profile)    │
├─────────────────────────────────────────────────────────────┤
│                    感知层 (Perception Layer)                 │
├─────────────────────────────────────────────────────────────┤
│  人脸识别  │  声纹识别  │  表情识别  │  手势识别  │  语音处理  │
├─────────────────────────────────────────────────────────────┤
│                    模型层 (Model Layer)                      │
├─────────────────────────────────────────────────────────────┤
│  InsightFace │ SpeechBrain │   FER/DeepFace   │  MediaPipe  │
│              │             │                  │             │
│         Qwen 2.5 Omni (云端API)              │  本地模型    │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 技术栈

### 核心框架
- **Python 3.10+**: 主要开发语言
- **OpenCV**: 计算机视觉处理
- **NumPy/SciPy**: 数值计算和科学计算
- **scikit-learn**: 机器学习工具

### AI模型
- **Qwen 2.5 Omni**: 多模态对话模型 (云端)
- **InsightFace**: 人脸识别模型
- **SpeechBrain**: 语音处理和声纹识别
- **MediaPipe**: 手势和姿态识别
- **FER/DeepFace**: 表情识别

### 数据处理
- **Faiss**: 向量相似度搜索
- **Librosa**: 音频信号处理
- **MoviePy**: 视频处理
- **Cryptography**: 数据加密

## 📦 安装说明

### 环境要求
- Python 3.10+
- CUDA 11.8+ (可选，用于GPU加速)
- 4GB+ RAM
- 2GB+ 存储空间

### 快速安装

1. **克隆项目**
```bash
git clone https://github.com/luocf/multi_modal_assistant.git
cd multi_modal_assistant
```

2. **创建虚拟环境**
```bash
# 使用 conda (推荐)
conda create -n multi-modal python=3.10
conda activate multi-modal

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

3. **安装依赖**
```bash
# macOS 用户需要先安装 portaudio
brew install portaudio

# 安装 Python 依赖
pip install -r requirements.txt
```

4. **下载模型文件**
```bash
# 运行模型下载脚本
python setup.py
```

5. **配置API密钥**
```bash
# 设置 Qwen API 密钥
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 详细安装脚本
项目提供了多个安装脚本：
- `setup.sh`: Linux/macOS 自动安装
- `setup.bat`: Windows 自动安装
- `setup_conda.sh`: Conda 环境安装
- `install_dependencies.sh`: 仅安装依赖

## 🚀 使用方法

### 基本使用

1. **启动助手**
```bash
python app/main.py
```

2. **混合模式启动** (推荐)
```bash
./run_hybrid.sh
```

3. **API服务模式**
```bash
python app/api/server.py
```

### 配置选项

主要配置文件位于 `app/config/` 目录：

- `settings.py`: 全局设置
- `model_config.py`: 模型配置
- `audio_config.py`: 音频参数
- `user_config.py`: 用户管理配置
- `voice_config.py`: 语音识别配置

### 使用示例

```python
from app.core.architecture import MultiModalAssistant

# 初始化助手
assistant = MultiModalAssistant()

# 开始交互
assistant.start_interaction()

# 处理语音输入
response = assistant.process_voice_input("你好，今天天气怎么样？")

# 处理图像输入
user_info = assistant.process_image_input(image_data)
```

## 📊 功能演示

### 身份识别流程
1. **首次接触**: 系统自动检测新用户，创建临时身份
2. **特征学习**: 收集人脸、声纹等生物特征
3. **身份确认**: 多模态特征融合确认用户身份
4. **画像更新**: 根据交互历史更新用户偏好

### 对话交互示例
```
用户: [通过摄像头检测到熟悉面孔]
助手: "欢迎回来！根据您之前的偏好，今天为您推荐..."

用户: "今天心情不太好"
助手: [检测到沮丧表情] "我注意到您看起来有些疲惫，要不要听点轻松的音乐？"

用户: [做出挥手手势]
助手: "再见！期待下次见面！"
```

## 🧪 测试

### 运行测试套件
```bash
# 运行所有测试
python run_tests.py

# 运行特定测试
python -m pytest app/tests/test_identity_manager.py

# 性能测试
python app/benchmarks/benchmark_system.py
```

### 快速功能测试
```bash
# 快速系统测试
python quick_test.py

# 语音识别测试
python app/test_voice_recognition.py

# 云端模型测试
python test_cloud_model.py
```

## 📁 项目结构

```
multi_modal_assistant/
├── app/                          # 主应用目录
│   ├── api/                     # API 服务器
│   ├── config/                  # 配置文件
│   ├── core/                    # 核心功能模块
│   │   ├── perception/          # 感知模块
│   │   ├── identity/            # 身份管理
│   │   ├── dialogue/            # 对话管理
│   │   ├── user/               # 用户管理
│   │   └── enhancement/         # 增强功能
│   ├── models/                  # 模型管理
│   ├── tests/                   # 单元测试
│   └── main.py                  # 主入口
├── docs/                        # 文档
├── models/                      # 模型文件存储
├── requirements.txt             # 依赖列表
├── setup.py                     # 安装脚本
└── README.md                    # 项目说明
```

## 🔒 隐私与安全

- **本地处理**: 敏感的生物特征数据仅在本地处理
- **数据加密**: 用户画像采用AES加密存储
- **访问控制**: 支持多级权限管理
- **数据脱敏**: 可选的数据匿名化功能
- **审计日志**: 完整的操作记录和审计追踪

## 🛠️ 开发指南

### 添加新的感知模块
1. 在 `app/core/perception/` 创建新模块
2. 继承 `BasePerceptionModule` 基类
3. 实现 `process()` 和 `reset()` 方法
4. 在 `model_factory.py` 中注册新模块

### 扩展对话能力
1. 修改 `dialogue_manager.py` 添加新的对话逻辑
2. 更新用户画像字段定义
3. 调整响应生成策略

### 性能优化建议
- 使用GPU加速推理 (CUDA)
- 启用模型量化减少内存占用
- 配置合适的批处理大小
- 使用异步处理提高并发性能

## 📈 性能指标

### 基准测试结果 (测试环境: MacBook Pro M1, 16GB RAM)
- **人脸识别**: ~50ms/frame
- **声纹识别**: ~200ms/audio_clip
- **表情识别**: ~30ms/frame
- **对话响应**: ~800ms (含网络延迟)
- **内存占用**: ~2GB (所有模块加载)

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范
- 遵循 PEP 8 Python 代码风格
- 添加必要的文档字符串
- 编写单元测试
- 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Qwen Team](https://github.com/QwenLM/Qwen) - 提供强大的多模态对话模型
- [InsightFace](https://github.com/deepinsight/insightface) - 人脸识别技术
- [SpeechBrain](https://github.com/speechbrain/speechbrain) - 语音处理框架
- [MediaPipe](https://github.com/google/mediapipe) - 多媒体处理管道

## 📞 联系方式

- 项目主页: https://github.com/luocf/multi_modal_assistant
- 问题反馈: [GitHub Issues](https://github.com/luocf/multi_modal_assistant/issues)
- 邮箱: lcf06@163.com

---

⭐ 如果这个项目对您有帮助，请给个星标支持！ 