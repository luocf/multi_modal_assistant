#!/bin/bash

# 设置pip镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装依赖
pip install -e . --no-cache-dir

# 如果上面的命令失败，尝试使用阿里云镜像
if [ $? -ne 0 ]; then
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
    pip config set global.trusted-host mirrors.aliyun.com
    pip install -e . --no-cache-dir
fi

# 如果还是失败，尝试单独安装mtcnn
if [ $? -ne 0 ]; then
    pip install mtcnn -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install -e . --no-deps
fi 