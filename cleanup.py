#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
项目清理脚本
删除临时文件、缓存文件和其他无用文件
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_project():
    """清理项目中的无用文件"""
    print("🧹 开始清理项目...")
    
    # 要删除的文件模式
    patterns_to_delete = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        "**/.pytest_cache",
        "**/.coverage",
        "**/debug_audio.*",
        "**/test_voice.*",
        "**/temp_*",
        "**/*.tmp",
        "**/*.bak",
        "**/*.swp",
        "**/*.swo",
        "**/*~",
    ]
    
    # 要删除的目录
    directories_to_delete = [
        "moviepy_wheel",
        "temp",
        "tmp",
        ".pytest_cache",
        "htmlcov",
        "build",
        "dist",
        "*.egg-info",
    ]
    
    deleted_count = 0
    
    # 删除匹配模式的文件
    for pattern in patterns_to_delete:
        for path in Path(".").glob(pattern):
            try:
                if path.is_file():
                    path.unlink()
                    print(f"🗑️ 删除文件: {path}")
                    deleted_count += 1
                elif path.is_dir():
                    shutil.rmtree(path)
                    print(f"🗑️ 删除目录: {path}")
                    deleted_count += 1
            except Exception as e:
                print(f"❌ 删除失败 {path}: {e}")
    
    # 删除特定目录
    for dir_pattern in directories_to_delete:
        for path in Path(".").glob(dir_pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"🗑️ 删除目录: {path}")
                    deleted_count += 1
            except Exception as e:
                print(f"❌ 删除失败 {path}: {e}")
    
    # 清理空目录
    for root, dirs, files in os.walk(".", topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # 空目录
                    os.rmdir(dir_path)
                    print(f"🗑️ 删除空目录: {dir_path}")
                    deleted_count += 1
            except Exception as e:
                pass  # 忽略无法删除的目录
    
    print(f"✅ 清理完成！共删除 {deleted_count} 个文件/目录")
    
    # 显示项目大小
    total_size = sum(f.stat().st_size for f in Path(".").rglob("*") if f.is_file())
    print(f"📊 项目总大小: {total_size / 1024 / 1024:.2f} MB")

def show_project_structure():
    """显示清理后的项目结构"""
    print("\n📁 项目结构:")
    
    important_dirs = [
        "app/config",
        "app/core",
        "app/models",
        "docs",
        "models"
    ]
    
    for dir_path in important_dirs:
        if os.path.exists(dir_path):
            print(f"📂 {dir_path}/")
            for item in sorted(os.listdir(dir_path)):
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    print(f"   📁 {item}/")
                else:
                    print(f"   📄 {item}")

if __name__ == "__main__":
    cleanup_project()
    show_project_structure() 