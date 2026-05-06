from __future__ import annotations

"""测试验证 Agent

执行流程：
1. LLM 基于完整代码文件内容生成测试代码
2. 将测试文件写入 project_path（由 PipelineService 负责）
3. 用 subprocess 真实运行 pytest，解析实际结果
4. 若 pytest 不可用或执行失败，回退到 LLM 模拟结果并标记 needs_human_review=True
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base_agent import BaseAgent, AgentInput, AgentOutput

# pytest 单次运行的最长等待时间（秒）
PYTEST_TIMEOUT_SECONDS = 120
# 传给 LLM 的单文件最大字符数（防止超长 prompt；None = 不截断）
CODE_CONTEXT_MAX_CHARS: Optional[int] = None


class TestAgent(BaseAgent):
    """
    测试验证 Agent
    职责：为生成的代码编写测试，真实执行 pytest，输出可信测试报告
    """

    async def execute(self, input_data: AgentInput) -> AgentOutput:
        code_files: Dict[str, str] = input_data.context.get("generated_code", {}) or {}
        project_path: str = input_data.context.get("project_path", "") or ""
        feedback: Optional[str] = input_data.human_feedback

        # ── 1. LLM 生成测试代码 ─────────────────────────────────────────
        test_files, llm_results, token_usage, model = await self._llm_generate_tests(
            code_files=code_files,
            feedback=feedback,
        )

        # ── 2. 真实运行 pytest ──────────────────────────────────────────
        real_results, pytest_ran = self._run_pytest(
            project_path=project_path,
            test_files=test_files,
        )

        # 以真实结果为准；如果 pytest 没跑起来，使用 LLM 模拟结果并标记需审批
        final_results = real_results if pytest_ran else llm_results
        needs_review = (not pytest_ran) or (not self._all_passed(final_results))

        report = self._format_report(final_results, pytest_ran=pytest_ran)
        passed = final_results.get("passed", 0)
        total = final_results.get("total", 0)

        return AgentOutput(
            success=self._all_passed(final_results),
            result={
                "test_files": test_files,
                "test_results": final_results,
                "report": report,
                "pass_rate": f"{passed}/{total}",
                "pytest_executed": pytest_ran,
            },
            summary=(
                f"测试完成（{'真实执行' if pytest_ran else 'LLM 模拟'}）：{passed}/{total} 通过"
            ),
            details=report,
            needs_human_review=needs_review,
            token_usage=token_usage,
            model=model,
        )

    # ------------------------------------------------------------------
    # LLM 生成测试
    # ------------------------------------------------------------------

    async def _llm_generate_tests(
        self,
        *,
        code_files: Dict[str, str],
        feedback: Optional[str],
    ) -> Tuple[Dict[str, str], dict, Optional[dict], str]:
        """调用 LLM 生成测试代码。

        返回 (test_files, llm_results, token_usage, model)
        """
        code_context = self._build_code_context(code_files)

        user_message = f"""请为以下代码编写全面的单元测试。

## 源代码文件
{code_context}

## 测试要求
1. 使用 pytest 框架
2. 对 FastAPI 应用使用 httpx.AsyncClient 或 TestClient
3. 覆盖：正常路径、边界条件、错误处理
4. 每个关键模块至少 3 个测试用例
5. 测试文件路径以 `tests/` 开头，命名为 `test_<模块名>.py`
6. 测试中不要依赖真实数据库或外部服务——使用 mock 或 fixture

## 输出格式
每个测试文件用以下格式包裹（不要用 Markdown 代码块）：
<file path="tests/test_xxx.py">
<content>
pytest 测试代码（完整可运行）
</content>
</file>

所有文件输出完后，输出一个 JSON 摘要：
<test_summary>
{{"total": N, "passed": N, "failed": 0, "coverage": 80, "errors": []}}
</test_summary>

约束：
- 测试代码必须语法正确，可直接运行
- 不要输出任何解释或 Markdown
- import 路径必须与源代码文件实际路径一致
"""
        if feedback:
            user_message += f"\n\n根据以下反馈调整测试：\n{feedback}"

        llm_response = await self.call_llm_response(user_message)
        response = llm_response.content

        test_files = self._parse_tagged_test_files(response)
        llm_results = self._parse_test_summary(response)

        # 兜底：没解析到任何测试文件时，把 LLM 输出包成一个占位文件
        if not test_files:
            test_files = {
                "tests/test_generated.py": (
                    "# LLM 生成的测试（未能解析标准格式，请人工确认）\n"
                    f'"""\n{response[:2000]}\n"""\n'
                )
            }

        return test_files, llm_results, llm_response.usage, llm_response.model or self.model_name

    def _build_code_context(self, code_files: Dict[str, str]) -> str:
        """把源代码文件拼成可读块。CODE_CONTEXT_MAX_CHARS 控制截断。"""
        if not code_files:
            return "（未提供源代码文件）"
        parts: List[str] = []
        for path, content in code_files.items():
            if CODE_CONTEXT_MAX_CHARS and len(content) > CODE_CONTEXT_MAX_CHARS:
                body = content[:CODE_CONTEXT_MAX_CHARS] + f"\n... （截断，共 {len(content)} 字符）"
            else:
                body = content
            parts.append(f"=== {path} ===\n{body}")
        return "\n\n".join(parts)

    def _parse_tagged_test_files(self, response: str) -> Dict[str, str]:
        """解析 <file path="..."> <content>...</content> </file> 格式。"""
        files: Dict[str, str] = {}
        pattern = re.compile(
            r"<file\s+path=[\"']([^\"']+)[\"']>\s*"
            r"(?:<summary>.*?</summary>\s*)?"
            r"<content>\s*(.*?)\s*</content>\s*</file>",
            re.DOTALL,
        )
        for match in pattern.finditer(response):
            path = match.group(1).strip()
            content = match.group(2)
            if path:
                files[path] = content
        return files

    def _parse_test_summary(self, response: str) -> dict:
        """解析 <test_summary> JSON 块；失败时返回全零结果。"""
        defaults = {"total": 0, "passed": 0, "failed": 0, "coverage": 0, "errors": []}
        pattern = re.compile(r"<test_summary>\s*(.*?)\s*</test_summary>", re.DOTALL)
        match = pattern.search(response)
        if not match:
            # 兼容旧格式：```json ... ```
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                match_text = json_match.group(1)
            else:
                return defaults
        else:
            match_text = match.group(1)
        try:
            parsed = json.loads(match_text)
            if isinstance(parsed, dict):
                defaults.update(parsed)
        except Exception:
            pass
        return defaults

    # ------------------------------------------------------------------
    # 真实 pytest 执行
    # ------------------------------------------------------------------

    def _run_pytest(
        self,
        *,
        project_path: str,
        test_files: Dict[str, str],
    ) -> Tuple[dict, bool]:
        """在 project_path 下写入测试文件并运行 pytest。

        返回 (results_dict, pytest_ran_successfully)
        """
        if not project_path or not Path(project_path).is_dir():
            return self._empty_results("project_path 不存在，跳过真实执行"), False

        base_dir = Path(project_path)
        written: List[Path] = []

        try:
            # 写入测试文件
            for rel_path, content in test_files.items():
                target = base_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(target)

            if not written:
                return self._empty_results("没有可写入的测试文件"), False

            # 运行 pytest
            python_exe = self._find_python(base_dir)
            cmd = [
                python_exe, "-m", "pytest",
                "tests/",
                "-v",
                "--tb=short",
                "--no-header",
                "-q",
                "--timeout=30",  # 需要 pytest-timeout；没装则被忽略
            ]
            completed = subprocess.run(
                cmd,
                cwd=str(base_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=PYTEST_TIMEOUT_SECONDS,
                env={**os.environ, "PYTHONPATH": str(base_dir)},
            )
            results = self._parse_pytest_output(
                completed.stdout + completed.stderr,
                returncode=completed.returncode,
            )
            return results, True

        except FileNotFoundError:
            return self._empty_results("找不到 python/pytest，跳过真实执行"), False
        except subprocess.TimeoutExpired:
            return self._empty_results(f"pytest 超时（>{PYTEST_TIMEOUT_SECONDS}s）"), False
        except Exception as exc:
            return self._empty_results(f"pytest 执行异常：{exc}"), False

    def _find_python(self, base_dir: Path) -> str:
        """优先使用 venv/虚拟环境里的 python，回退到当前解释器。"""
        for candidate in (
            base_dir / "venv" / "bin" / "python",
            base_dir / ".venv" / "bin" / "python",
            base_dir / "venv" / "Scripts" / "python.exe",
            base_dir / ".venv" / "Scripts" / "python.exe",
        ):
            if candidate.exists():
                return str(candidate)
        return sys.executable

    def _parse_pytest_output(self, output: str, *, returncode: int) -> dict:
        """从 pytest -q 输出解析 passed / failed / error 数量。

        pytest 退出码：0 = 全部通过，1 = 有失败，2 = 中断，3-5 = 内部错误
        """
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "coverage": 0,
            "raw_output": output[-3000:],  # 保留最后 3000 字符用于调试
        }

        # 格式 1：" 3 passed, 1 failed in 0.42s"
        summary_match = re.search(
            r"(\d+) passed(?:,\s*(\d+) failed)?(?:,\s*(\d+) error)?",
            output,
        )
        if summary_match:
            results["passed"] = int(summary_match.group(1) or 0)
            results["failed"] = int(summary_match.group(2) or 0)
            error_count = int(summary_match.group(3) or 0)
            results["total"] = results["passed"] + results["failed"] + error_count
            return results

        # 格式 2：只有失败 " 2 failed in 0.31s"
        failed_only = re.search(r"(\d+) failed", output)
        if failed_only:
            results["failed"] = int(failed_only.group(1))
            results["total"] = results["failed"]
            return results

        # 格式 3：收集到 0 项测试（no tests found）
        if "no tests ran" in output.lower() or "collected 0 items" in output.lower():
            results["errors"] = ["未收集到测试用例，请检查测试文件路径"]
            return results

        # 兜底：根据退出码推断
        if returncode == 0:
            results["passed"] = 1
            results["total"] = 1
        else:
            results["failed"] = 1
            results["total"] = 1

        # 提取 FAILED 行作为错误摘要
        fail_lines = [
            line.strip()
            for line in output.splitlines()
            if line.startswith("FAILED") or "ERROR" in line
        ]
        results["errors"] = fail_lines[:10]

        return results

    @staticmethod
    def _empty_results(reason: str) -> dict:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": [reason],
            "coverage": 0,
        }

    @staticmethod
    def _all_passed(results: dict) -> bool:
        total = results.get("total", 0)
        passed = results.get("passed", 0)
        return total > 0 and passed == total

    # ------------------------------------------------------------------
    # 报告格式化
    # ------------------------------------------------------------------

    def _format_report(self, results: dict, *, pytest_ran: bool) -> str:
        mode = "真实 pytest 执行" if pytest_ran else "LLM 模拟（pytest 未运行）"
        lines = [
            "## 测试报告",
            "",
            f"- **执行方式**: {mode}",
            f"- **总计**: {results['total']} 项测试",
            f"- **通过**: {results['passed']} ✅",
            f"- **失败**: {results['failed']} ❌",
            f"- **代码覆盖率**: {results.get('coverage', 'N/A')}%",
        ]
        if results.get("errors"):
            lines.append("\n### 错误详情")
            for err in results["errors"]:
                lines.append(f"- {err}")
        if not pytest_ran:
            lines.extend([
                "",
                "> ⚠️ pytest 未实际运行，以上结果为 LLM 估算。",
                "> 建议手动运行 `pytest tests/` 验证。",
            ])
        return "\n".join(lines)
