import os
import sys
import pytest

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """设置测试环境"""
    # 这里可以添加其他测试环境设置
    yield
    # 测试完成后的清理工作 