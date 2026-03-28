"""
ProgressBar — 终端底部独占一行进度条显示
支持简单ASCII进度条、居中显示、步骤间更新
与现有logger颜色系统兼容
"""

import os
import sys
from typing import Optional

# 尝试导入颜色常量，如果不可用则使用空字符串
try:
    from .logger import (
        COLOR_256_TEAL, COLOR_BRIGHT_GREEN, COLOR_CYAN, COLOR_256_ORANGE,
        COLOR_RESET
    )
    # 阶段颜色映射
    PHASE_COLORS = {
        1: COLOR_BRIGHT_GREEN,  # 亮绿色 - Phase1
        2: COLOR_BRIGHT_GREEN,  # 亮绿色 - Phase2
        3: COLOR_BRIGHT_GREEN,  # 亮绿色 - Phase3
        4: COLOR_BRIGHT_GREEN,  # 亮绿色 - Phase4
    }
    COLOR_PROGRESS_BAR = COLOR_BRIGHT_GREEN
    COLOR_PROGRESS_TEXT = COLOR_BRIGHT_GREEN
except ImportError:
    # 降级处理：无颜色
    COLOR_256_TEAL = COLOR_BRIGHT_GREEN = COLOR_CYAN = COLOR_256_ORANGE = ""
    COLOR_RESET = ""
    PHASE_COLORS = {1: "", 2: "", 3: "", 4: ""}
    COLOR_PROGRESS_BAR = ""
    COLOR_PROGRESS_TEXT = ""


class ProgressBar:
    """
    简单进度条，显示在终端底部独占一行

    示例输出：
                            [#####...............] 38% (5/13)
    """

    def __init__(self, total_steps: int, step_descriptions: Optional[list] = None):
        """
        初始化进度条

        Args:
            total_steps: 总步骤数
            step_descriptions: 步骤描述列表（可选）
        """
        self.total_steps = total_steps
        self.step_descriptions = step_descriptions or []
        self.current_step = 0  # 当前步骤索引（0-based）
        self._terminal_width = 80  # 默认终端宽度
        self._terminal_height = 24  # 默认终端高度
        self._update_terminal_size()

        # 检查NO_COLOR环境变量
        self.no_color = os.environ.get("NO_COLOR", "").strip().lower() not in ("", "0", "false")

        # 检测终端是否支持ANSI转义码（光标控制和颜色）
        # 要求：标准输出是终端，且未设置NO_COLOR
        self.supports_ansi = (
            os.isatty(sys.stdout.fileno())
            and not self.no_color
        )

    def _update_terminal_size(self):
        """获取当前终端尺寸（宽度和高度）"""
        try:
            import shutil
            size = shutil.get_terminal_size()
            self._terminal_width = size.columns
            self._terminal_height = size.lines
            # 确保最小宽度
            if self._terminal_width < 40:
                self._terminal_width = 40
        except (AttributeError, OSError, ImportError):
            # 降级处理
            self._terminal_width = 80
            self._terminal_height = 24

    def update(self, step_index: int, step_id: str):
        """
        更新进度条并显示在底部独占一行

        Args:
            step_index: 步骤索引（0-based），-1表示开始前
            step_id: 步骤ID，如 "step_01"
        """
        # 确保索引在有效范围内
        if step_index < -1:
            step_index = -1
        elif step_index >= self.total_steps:
            step_index = self.total_steps - 1

        self.current_step = step_index
        self._render(step_id)

    def _render(self, step_id: str):
        """渲染进度条到终端底部独占一行"""
        # 计算进度（当前步骤索引+1 / 总步骤数）
        # 如果step_index为-1（开始前），进度为0
        progress = (self.current_step + 1) / self.total_steps if self.current_step >= 0 else 0.0
        progress = max(0.0, min(1.0, progress))  # 限制在0-1之间

        # 获取终端宽度
        self._update_terminal_size()

        # 每边预留的空格数（控制间隙大小）
        side_margin = 0

        # 先计算百分比和步骤计数（不依赖进度条宽度）
        percentage = int(progress * 100)
        step_count = f"{self.current_step + 1}/{self.total_steps}" if self.current_step >= 0 else f"0/{self.total_steps}"

        # 计算非进度条部分的实际长度：方括号2字符 + 空格 + 百分比 + 空格 + 步骤计数
        non_bar_length = 2 + len(f" {percentage}% ({step_count})")

        # 计算可用的进度条宽度：终端宽度减去两边边距和非进度条部分
        available_width = self._terminal_width - 2 * side_margin
        # 计算最大进度条宽度：可用宽度的98%，确保留有一些边距
        max_bar_width = int(available_width * 0.98) - non_bar_length
        bar_width = max(20, min(available_width - non_bar_length, max_bar_width))  # 最小20字符

        # 确保不超过可用宽度
        if bar_width + non_bar_length > available_width:
            bar_width = max(10, available_width - non_bar_length)

        # 构建进度条字符串
        bar_filled = int(bar_width * progress)
        bar_empty = bar_width - bar_filled
        bar_str = f"[{'#' * bar_filled}{'.' * bar_empty}]"

        # 确定当前阶段（从step_id推断，如step_01属于phase1）
        try:
            step_num = int(step_id.split('_')[1]) if step_id.startswith('step_') else 1
            # 步骤1-4: phase1, 5-8: phase2, 9-10: phase3, 11-13: phase4
            if 1 <= step_num <= 4:
                phase = 1
            elif 5 <= step_num <= 8:
                phase = 2
            elif 9 <= step_num <= 10:
                phase = 3
            elif 11 <= step_num <= 13:
                phase = 4
            else:
                phase = 1
        except (IndexError, ValueError):
            phase = 1

        # 组合完整字符串
        progress_text = f"{bar_str} {percentage}% ({step_count})"
        text_width = len(progress_text)  # 无颜色文本宽度

        # 计算填充：真正居中显示，确保最小边距
        total_needed_width = text_width + 2 * side_margin

        if total_needed_width <= self._terminal_width:
            # 终端足够宽，计算居中位置
            total_padding = self._terminal_width - text_width
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            # 确保最小边距
            if left_padding < side_margin:
                left_padding = side_margin
                right_padding = self._terminal_width - text_width - side_margin
        else:
            # 终端不够宽，左对齐显示
            left_padding = side_margin
            right_padding = self._terminal_width - text_width - side_margin

        # 构建无颜色完整行（确保独占一整行）
        plain_line = ' ' * left_padding + progress_text + ' ' * right_padding
        # 确保长度正好等于终端宽度（处理可能的差1情况）
        if len(plain_line) < self._terminal_width:
            plain_line += ' ' * (self._terminal_width - len(plain_line))
        elif len(plain_line) > self._terminal_width:
            plain_line = plain_line[:self._terminal_width]

        # 构建最终输出文本（可能带颜色）
        if self.supports_ansi:
            phase_color = PHASE_COLORS.get(phase, COLOR_PROGRESS_BAR)
            # 只对进度文本部分添加颜色，填充空格保持无色
            colored_part = f"{phase_color}{progress_text}{COLOR_RESET}"
            final_text = ' ' * left_padding + colored_part + ' ' * right_padding
            # 确保长度正确
            if len(final_text) < self._terminal_width:
                final_text += ' ' * (self._terminal_width - len(final_text))
        else:
            final_text = plain_line

        # 总是独占一行显示（简化逻辑，确保进度条独占一整行）
        # 输出进度条，后跟换行符
        sys.stdout.write(final_text + "\n")
        sys.stdout.flush()

    def clear(self):
        """清除进度条显示（可选）"""
        # 对于简单的步骤间更新，不需要清除
        pass

    def complete(self, step_id: Optional[str] = None):
        """标记进度条完成"""
        self.current_step = self.total_steps - 1
        if step_id is None:
            step_id = f"step_{self.total_steps:02d}"
        self._render(step_id)