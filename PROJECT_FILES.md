# 项目文件说明

## 📁 目录结构和文件用途

### ✅ 核心文件（必需保留）

#### 应用程序核心
```
app/
├── config/                     # 配置文件目录
│   ├── audio_config.py        # 音频配置
│   ├── enhancement_config.py  # 增强功能配置
│   ├── model_config.py        # 模型配置
│   ├── user_config.py         # 用户管理配置
│   └── voice_config.py        # 声音配置
├── core/                      # 核心功能模块
│   ├── architecture.py        # 系统架构和主要逻辑
│   ├── conversation/          # 对话管理模块
│   ├── dialogue/              # 对话生成模块
│   ├── enhancement/           # 功能增强模块
│   ├── identity/              # 身份识别模块
│   ├── monitoring/            # 系统监控模块
│   ├── perception/            # 感知模块
│   │   ├── emotion_recognizer.py  # 表情识别
│   │   ├── face_recognizer.py     # 人脸识别
│   │   ├── gesture_recognizer.py  # 手势识别
│   │   ├── lip_analyzer.py        # 唇语分析
│   │   └── voice_recognizer.py    # 声纹识别
│   ├── user/                  # 用户管理模块
│   └── utils.py               # 工具函数
└── models/                    # 模型相关
    ├── base_model.py          # 基础模型类
    ├── model_factory.py       # 模型工厂
    ├── qwen_cloud_model.py    # 云端Qwen模型
    └── qwen_model.py          # 本地Qwen模型
```

#### 主程序文件
```
main.py                        # 主程序入口
requirements.txt               # Python依赖列表
```

#### 配置和脚本
```
setup_cloud_dialogue.py        # 云端对话配置脚本
run.sh                         # 标准运行脚本
run_hybrid.sh                  # 混合模式运行脚本
cleanup.py                     # 项目清理脚本
```

### 🔄 开发和测试文件（开发时有用）

#### 测试文件
```
app/tests/                     # 单元测试目录
app/benchmarks/               # 性能测试
run_tests.py                  # 测试运行器
test_cloud_model.py           # 云端模型测试
quick_test.py                 # 快速测试
```

#### API接口（可选）
```
app/api/                      # Web API接口
```

### 📚 文档和数据

#### 文档目录
```
docs/                         # 项目文档
PROJECT_FILES.md              # 本文件
README.md                     # 项目说明
```

#### 数据目录
```
models/                       # 模型文件存储
data/                         # 用户数据（.gitignore中排除）
```

### ❌ 无用文件（应删除或忽略）

#### Python缓存文件
```
__pycache__/                  # Python字节码缓存
*.pyc, *.pyo, *.pyd          # 编译的Python文件
```

#### 临时文件
```
debug_audio.wav               # 调试音频文件
test_voice.wav                # 测试音频文件
moviepy_wheel/                # 临时安装包
temp/, tmp/                   # 临时目录
*.tmp, *.bak                  # 临时和备份文件
```

#### 测试数据
```
test_identities/              # 测试身份数据
user_profiles.enc             # 加密的用户画像（隐私数据）
```

## 🛠️ 维护建议

### 定期清理
```bash
# 运行清理脚本
python cleanup.py

# 手动清理Python缓存
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### Git版本控制
- ✅ **应该提交**: 源代码、配置文件、文档、脚本
- ❌ **不应提交**: 缓存文件、临时文件、大型模型文件、用户数据

### 文件大小管理
- **大型模型文件**: 使用Git LFS或外部存储
- **用户数据**: 本地存储，不提交到版本控制
- **日志文件**: 定期清理或轮转

## 📊 项目大小优化

### 减少项目大小
1. 删除不必要的测试文件
2. 清理Python缓存
3. 移除临时文件
4. 压缩或外部存储大型模型

### 推荐的项目结构
```
multi_model_demo/             # 根目录
├── .gitignore               # Git忽略文件
├── main.py                  # 主程序
├── requirements.txt         # 依赖列表
├── cleanup.py               # 清理脚本
├── app/                     # 应用核心
├── docs/                    # 文档
└── models/                  # 模型（大文件用.gitignore排除）
```

这样的结构既保持了功能完整性，又避免了不必要的文件膨胀。 