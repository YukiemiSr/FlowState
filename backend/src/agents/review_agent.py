from __future__ import annotations

"""代码评审 Agent"""

import json
from .base_agent import BaseAgent, AgentInput, AgentOutput
from typing import Dict


class ReviewAgent(BaseAgent):
    """
    代码评审 Agent
    职责：审查代码质量、安全性、最佳实践，输出评审报告
    """

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        code_files = input_data.context.get("generated_code", {})
        test_report = input_data.context.get("test_report", "")
        code_diff = input_data.context.get("code_diff")  # git diff 注入
        feedback = input_data.human_feedback

        review, token_usage, model = await self._llm_review(
            code_files, test_report, feedback, code_diff=code_diff
        )

        issues = review.get("issues", [])
        critical = sum(1 for i in issues if i["severity"] == "critical")

        return AgentOutput(
            success=critical == 0,
            result={
                "review": review,
                "report": self._format_review(review),
                "issues": issues,
                "score": review.get("score", 0),
            },
            summary=f"评审完成：评分 {review.get('score', 0)}/100，发现 {len(issues)} 个问题",
            details=self._format_review(review),
            needs_human_review=True,
            token_usage=token_usage,
            model=model,
        )

    async def _llm_review(
        self,
        code_files: Dict[str, str],
        test_report: str,
        feedback: str | None,
        *,
        code_diff: str | None = None,
    ) -> tuple[dict, dict[str, int] | None, str]:
        """调用 LLM 进行代码评审。

        优先使用 git diff 作为评审输入（更聚焦于本次变更）；
        diff 不可用时回退到全量代码内容（每文件最多 1200 字符）。
        """
        if code_diff:
            code_context = f"## Git Diff（本次变更）\n```diff\n{code_diff}\n```"
        else:
            code_context = "## 代码文件\n" + "\n\n".join(
                f"=== {path} ===\n{content[:1200]}"
                for path, content in code_files.items()
            )

        user_message = f"""请对以下代码变更进行全面评审。

{code_context}

## 测试报告
{test_report or "（未提供测试报告）"}

请输出 JSON 格式的评审报告，包含以下字段：
1. score: 0-100 的评分
2. summary: 总体评价
3. strengths: 优点列表
4. issues: 问题列表（每个问题包含 file, line, severity, message）
   - 重点关注：正确性、安全漏洞、性能问题、边界条件缺失
5. suggestions: 改进建议列表

严重级别：critical（阻断）/ high（重要）/ medium（一般）/ low（轻微）

只输出 JSON，不要其他内容。
"""

        if feedback:
            user_message += f"\n\n根据以下反馈调整评审：\n{feedback}"

        llm_response = await self.call_llm_response(user_message, expect_json=True)
        response = llm_response.content
        try:
            return json.loads(response), llm_response.usage, llm_response.model
        except json.JSONDecodeError:
            return {
                "score": 70,
                "summary": "评审完成（解析回退）",
                "strengths": ["代码结构完整"],
                "issues": [{"file": "unknown", "line": 0, "severity": "medium", "message": "请人工审查代码质量"}],
                "suggestions": ["建议人工审查"],
            }, llm_response.usage, llm_response.model

    def _format_review(self, review: dict) -> str:
        lines = [
            "## 代码评审报告",
            "",
            f"**评分: {review['score']}/100**",
            f"**总体评价: {review['summary']}**",
            "",
            "### ✅ 优点",
        ]
        for s in review["strengths"]:
            lines.append(f"- {s}")

        lines.append("\n### ⚠️ 问题清单")
        for issue in review["issues"]:
            sev_icon = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🔵",
            }
            icon = sev_icon.get(issue["severity"], "⚪")
            lines.append(
                f"- {icon} [{issue['severity']}] {issue['file']}:{issue['line']} "
                f"— {issue['message']}"
            )

        lines.append("\n### 💡 改进建议")
        for s in review["suggestions"]:
            lines.append(f"- {s}")

        return "\n".join(lines)
