"""
critic/revision_mixin.py
-------------------------
可选 Mixin：让 Phase Agent 在重新执行时感知 Critic 下发的 RevisionPlan。

使用方式（按需引入，不修改现有 Phase Agent 文件）：

    # 方式一：不使用 Mixin（现有代码完全不变）
    # Phase Agent 照常运行，Critic 检查结果决定是否重试
    # 缺点：LLM 重新执行时不知道上次哪里错了

    # 方式二：使用 Mixin（推荐，在不修改现有文件的前提下扩展）
    from critic.revision_mixin import RevisionAwareMixin
    class Phase2ServiceAgentV2(RevisionAwareMixin, Phase2ServiceAgent):
        def _step05_refactor_code(self):
            plan = self.get_revision_context("step_05")
            if plan:
                # 将修改指令追加到 LLM prompt 中
                extra = self._format_revision_context(plan)
                # ... 在 user_prompt 末尾加上 extra
            return super()._step05_refactor_code()

设计原则：
- 完全可选，不引入则现有 Phase Agent 行为不变
- 不修改任何现有文件
- 通过 StateStore 读取 RevisionPlan（已由 CriticAgent 写入）
"""

from __future__ import annotations

from typing import Optional

from critic.critic_agent import RevisionPlan


class RevisionAwareMixin:
    """
    Mixin：为 Phase Agent 提供读取 RevisionPlan 的能力。

    要求混入的类拥有 self.state: StateStore 属性（Phase Agent 均满足）。
    """

    def get_revision_context(self, step_id: str) -> Optional[RevisionPlan]:
        """
        获取 Critic 针对本步骤的最新修改计划。
        若不存在（首次执行）则返回 None。
        """
        data = self.state.get(f"revision_plan_{step_id}")
        if not data:
            return None
        return RevisionPlan(
            step_id=data["step_id"],
            failed_checks=data["failed_checks"],
            instructions=data["instructions"],
            context=data.get("context", {}),
            round_number=data.get("round_number", 1),
        )

    def _format_revision_context(self, plan: RevisionPlan) -> str:
        """
        将 RevisionPlan 格式化为可追加到 LLM prompt 末尾的文字段落。

        示例输出：
            ===== Critic 修改要求（第2轮）=====
            上次执行未通过以下检查项：
            - 四个标准函数均已定义
            - 重构代码运行验证通过

            修改指令：
            1. 重新生成 single_inference_refactor.py，确保包含所有四个函数
            2. 修复运行时错误：NameError: name 'model' is not defined
            =====================================
        """
        lines = [
            f"\n===== Critic 修改要求（第 {plan.round_number} 轮）=====",
            "上次执行未通过以下检查项：",
        ]
        for name in plan.failed_checks:
            detail = plan.context.get(name, "")
            lines.append(f"  - {name}" + (f": {detail}" if detail else ""))

        if plan.instructions:
            lines.append("\n修改指令：")
            for i, inst in enumerate(plan.instructions, 1):
                lines.append(f"  {i}. {inst}")

        lines.append("=" * 37)
        return "\n".join(lines)

    def build_revision_aware_prompt(self, step_id: str, base_prompt: str) -> str:
        """
        便捷方法：在 base_prompt 末尾追加修改计划（如有）。
        Phase Agent 的 _stepXX 方法可直接调用此方法构建完整 prompt。

        示例：
            user_prompt = self.build_revision_aware_prompt(
                "step_05",
                REFACTOR_USER_TEMPLATE.format(original_code=original_code)
            )
        """
        plan = self.get_revision_context(step_id)
        if plan is None:
            return base_prompt
        return base_prompt + self._format_revision_context(plan)
