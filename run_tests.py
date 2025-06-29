#!/usr/bin/env python3
import os
import sys
import unittest
import pytest

def run_tests():
    """运行所有测试"""
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # 使用unittest发现并运行测试
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('app/tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 使用pytest运行测试并生成覆盖率报告
    pytest.main(['app/tests', '--cov=app', '--cov-report=term-missing'])
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 