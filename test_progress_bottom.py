#!/usr/bin/env python3
"""
测试底部进度条显示
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.progress_bar import ProgressBar

def test_bottom_progress():
    """测试底部进度条显示"""
    print("测试进度条底部显示")
    print("=" * 80)
    print("日志行 1: 这是模拟日志输出")
    print("日志行 2: 进度条应该显示在底部独占一行")

    # 创建进度条
    descriptions = [f"步骤 {i+1}" for i in range(13)]
    pb = ProgressBar(total_steps=13, step_descriptions=descriptions)

    # 测试初始状态
    print("\n初始化进度条 (step_index=-1):")
    pb.update(-1, "starting")

    time.sleep(0.5)
    print("\n日志行 3: 模拟步骤执行...")

    # 测试各个步骤
    for i in range(13):
        step_id = f"step_{i+1:02d}"
        print(f"\n执行步骤 {step_id}...")
        pb.update(i, step_id)
        time.sleep(0.1)

    # 测试完成状态
    print("\nPipeline完成:")
    pb.complete("step_13")

    print("\n测试完成！进度条应该在每步更新时显示在单独一行。")

if __name__ == "__main__":
    test_bottom_progress()