"""Git 命令封装服务。

设计原则：
- 纯命令封装，无业务语义；上层 PipelineService 决定何时调用。
- 全部走 subprocess.run(["git", ...])，不引入 GitPython 等第三方库。
- 命令出错统一抛 GitError（含子类）；调用方按需 catch。
- 路径参数统一接受 str | os.PathLike，内部转 Path。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

# ---------------------------------------------------------------------------
# 异常体系
# ---------------------------------------------------------------------------


class GitError(RuntimeError):
    """所有 git 操作失败的基类。"""

    def __init__(self, message: str, *, stderr: str | None = None, returncode: int | None = None) -> None:
        super().__init__(message)
        self.stderr = stderr or ""
        self.returncode = returncode


class GitNotInstalledError(GitError):
    """系统 PATH 中找不到 git 可执行文件。"""


class NotARepoError(GitError):
    """目标目录不是 git 仓库。"""


class NestedRepoError(GitError):
    """目标目录的父级已经是 git 仓库，禁止在内部 init。"""


class WorktreeBusyError(GitError):
    """worktree 路径已被占用或已存在残留。"""


# ---------------------------------------------------------------------------
# 公共类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GitCommandResult:
    stdout: str
    stderr: str
    returncode: int


DEFAULT_TIMEOUT_SECONDS = 30

DEFAULT_GITIGNORE_LINES = (
    "# FlowState 默认忽略",
    ".flowstate/",
    "",
    "# Python",
    "__pycache__/",
    "*.pyc",
    ".venv/",
    "venv/",
    "",
    "# Node",
    "node_modules/",
    "dist/",
    "build/",
    "",
    "# IDE / OS",
    ".idea/",
    ".vscode/",
    ".DS_Store",
    "",
    "# 环境变量",
    ".env",
    ".env.local",
)


# ---------------------------------------------------------------------------
# GitService
# ---------------------------------------------------------------------------


class GitService:
    """git CLI 的薄封装。无状态，可复用。"""

    def __init__(
        self,
        *,
        git_executable: str = "git",
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.git_executable = git_executable
        self.timeout = timeout

    # ------------------------------------------------------------------
    # 基础执行器
    # ------------------------------------------------------------------

    def _run(
        self,
        args: Iterable[str],
        *,
        cwd: Path,
        check: bool = True,
        env: dict | None = None,
    ) -> GitCommandResult:
        cmd = [self.git_executable, *args]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                env={**os.environ, **(env or {})},
                check=False,
            )
        except FileNotFoundError as exc:
            raise GitNotInstalledError(
                f"未找到 git 可执行文件: {self.git_executable}",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise GitError(
                f"git 命令超时（>{self.timeout}s）: {' '.join(cmd)}",
                stderr=str(exc),
            ) from exc

        if check and completed.returncode != 0:
            raise GitError(
                f"git 命令失败: {' '.join(cmd)}\n{completed.stderr.strip()}",
                stderr=completed.stderr,
                returncode=completed.returncode,
            )
        return GitCommandResult(
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            returncode=completed.returncode,
        )

    # ------------------------------------------------------------------
    # 仓库探测
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """git 可执行文件是否可用。"""
        try:
            self._run(["--version"], cwd=Path.cwd())
            return True
        except GitNotInstalledError:
            return False
        except GitError:
            return False

    def is_git_repo(self, path: str | os.PathLike) -> bool:
        """path 是否是 git 仓库根（含 .git 目录或 .git 文件）。"""
        target = Path(path)
        if not target.is_dir():
            return False
        return (target / ".git").exists()

    def find_repo_root(self, path: str | os.PathLike) -> Optional[Path]:
        """从 path 起向上递归找 .git，找到则返回该目录；未找到返回 None。"""
        target = Path(path).resolve()
        current = target
        while True:
            if (current / ".git").exists():
                return current
            if current.parent == current:
                return None
            current = current.parent

    def find_enclosing_repo(self, path: str | os.PathLike) -> Optional[Path]:
        """专用于嵌套仓库探测：从 path 的父目录开始向上找 .git。

        如果 path 自身就是 repo，返回 None（即没有"外层 repo"包住它）。
        """
        target = Path(path).resolve()
        if (target / ".git").exists():
            return None
        if target.parent == target:
            return None
        return self.find_repo_root(target.parent)

    # ------------------------------------------------------------------
    # 仓库初始化
    # ------------------------------------------------------------------

    def init_repo(
        self,
        path: str | os.PathLike,
        *,
        default_branch: str = "main",
    ) -> None:
        """在 path 处执行 git init。目录必须已存在。"""
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        # --initial-branch 在老版本 git 上不支持，先尝试新写法，失败 fallback
        try:
            self._run(
                ["init", f"--initial-branch={default_branch}"],
                cwd=target,
            )
        except GitError:
            self._run(["init"], cwd=target)
            # 老 git：手动改默认分支
            self._run(
                ["symbolic-ref", "HEAD", f"refs/heads/{default_branch}"],
                cwd=target,
            )

    def write_default_gitignore(
        self,
        path: str | os.PathLike,
        *,
        overwrite: bool = False,
    ) -> Path:
        """写入一份默认 .gitignore；存在则不动（除非 overwrite=True）。"""
        target = Path(path) / ".gitignore"
        if target.exists() and not overwrite:
            return target
        target.write_text("\n".join(DEFAULT_GITIGNORE_LINES) + "\n", encoding="utf-8")
        return target

    def ensure_gitignore_entry(
        self,
        path: str | os.PathLike,
        entry: str,
    ) -> None:
        """确保 .gitignore 中包含 entry；不存在则追加。"""
        gitignore = Path(path) / ".gitignore"
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
        lines = [line.strip() for line in existing.splitlines()]
        if entry.strip() in lines:
            return
        new_text = existing
        if existing and not existing.endswith("\n"):
            new_text += "\n"
        new_text += f"{entry}\n"
        gitignore.write_text(new_text, encoding="utf-8")

    def baseline_commit(
        self,
        path: str | os.PathLike,
        *,
        message: str = "chore: flowstate baseline",
        allow_empty: bool = True,
    ) -> str:
        """`git add . && git commit`；返回新 commit 的 sha。

        - 若工作区为空且 allow_empty=True，则以 --allow-empty 创建空提交。
        """
        target = Path(path)
        # 配置缺省 user.name / user.email（某些 CI 环境不带这俩）
        self._ensure_committer_identity(target)
        self._run(["add", "-A"], cwd=target)
        # 如果没有任何 staged 改动，且不允许空提交则报错
        has_changes = self._has_staged_changes(target)
        if not has_changes and not allow_empty:
            raise GitError("没有可提交的变更")

        commit_args = ["commit", "-m", message]
        if not has_changes:
            commit_args.append("--allow-empty")
        self._run(commit_args, cwd=target)
        return self.head_commit(target)

    def _ensure_committer_identity(self, repo: Path) -> None:
        """如果该仓库缺少 user.name / user.email，则注入一个 FlowState 默认值。"""
        for key, fallback in (
            ("user.name", "FlowState"),
            ("user.email", "flowstate@local"),
        ):
            try:
                result = self._run(
                    ["config", "--get", key],
                    cwd=repo,
                    check=False,
                )
            except GitError:
                result = None
            if not result or result.returncode != 0 or not result.stdout.strip():
                self._run(["config", key, fallback], cwd=repo)

    def _has_staged_changes(self, repo: Path) -> bool:
        result = self._run(
            ["diff", "--cached", "--quiet"],
            cwd=repo,
            check=False,
        )
        # `--quiet` 在有 staged 改动时返回码 1；无改动返回码 0
        return result.returncode == 1

    # ------------------------------------------------------------------
    # 分支与 worktree
    # ------------------------------------------------------------------

    def current_branch(self, repo: str | os.PathLike) -> str:
        """返回当前分支名。即使仓库还没有任何 commit 也能工作。"""
        # symbolic-ref 在空仓库（尚无 HEAD commit）下也能返回 refs/heads/<branch>
        result = self._run(
            ["symbolic-ref", "--short", "HEAD"],
            cwd=Path(repo),
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        # 兜底：使用 rev-parse（detached HEAD 等情况）
        result = self._run(["rev-parse", "--abbrev-ref", "HEAD"], cwd=Path(repo))
        return result.stdout.strip()

    def head_commit(self, repo: str | os.PathLike) -> str:
        result = self._run(["rev-parse", "HEAD"], cwd=Path(repo))
        return result.stdout.strip()

    def has_any_commit(self, repo: str | os.PathLike) -> bool:
        result = self._run(["rev-parse", "--verify", "HEAD"], cwd=Path(repo), check=False)
        return result.returncode == 0

    def add_worktree(
        self,
        repo: str | os.PathLike,
        worktree_path: str | os.PathLike,
        branch: str,
        *,
        base: str,
    ) -> None:
        """在 repo 仓库中新建 worktree，挂在 base 上的新分支 branch。

        失败场景：
        - worktree_path 已存在 → WorktreeBusyError
        - branch 已存在 → WorktreeBusyError
        """
        repo_path = Path(repo)
        wt_path = Path(worktree_path)
        if wt_path.exists():
            raise WorktreeBusyError(f"worktree 路径已存在: {wt_path}")

        wt_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._run(
                ["worktree", "add", str(wt_path), "-b", branch, base],
                cwd=repo_path,
            )
        except GitError as exc:
            stderr = exc.stderr.lower()
            if "already exists" in stderr or "is already checked out" in stderr:
                raise WorktreeBusyError(str(exc), stderr=exc.stderr) from exc
            raise

    def remove_worktree(
        self,
        repo: str | os.PathLike,
        worktree_path: str | os.PathLike,
        *,
        force: bool = True,
    ) -> None:
        """移除 worktree（默认 force，因为里面经常有未 commit 改动）。

        即使 git 命令失败，也尝试物理删除目录，确保 cleanup 幂等。
        """
        repo_path = Path(repo)
        wt_path = Path(worktree_path)
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(wt_path))
        try:
            self._run(args, cwd=repo_path, check=False)
        except GitError:
            pass
        # 物理兜底
        if wt_path.exists():
            shutil.rmtree(wt_path, ignore_errors=True)
        # 清理 stale worktree 元信息
        self._run(["worktree", "prune"], cwd=repo_path, check=False)

    def delete_branch(
        self,
        repo: str | os.PathLike,
        branch: str,
        *,
        force: bool = True,
    ) -> None:
        flag = "-D" if force else "-d"
        self._run(["branch", flag, branch], cwd=Path(repo), check=False)

    def list_worktrees(self, repo: str | os.PathLike) -> List[str]:
        result = self._run(["worktree", "list", "--porcelain"], cwd=Path(repo))
        paths: List[str] = []
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                paths.append(line[len("worktree ") :].strip())
        return paths

    # ------------------------------------------------------------------
    # 提交
    # ------------------------------------------------------------------

    def stage_all(self, worktree: str | os.PathLike) -> None:
        self._run(["add", "-A"], cwd=Path(worktree))

    def has_changes(self, worktree: str | os.PathLike) -> bool:
        """工作区或 staging 区是否有变更。"""
        target = Path(worktree)
        result = self._run(["status", "--porcelain"], cwd=target)
        return bool(result.stdout.strip())

    def commit(
        self,
        worktree: str | os.PathLike,
        message: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        """提交所有已 stage 的变更。返回新 commit 的 sha。

        若没有 staged 改动且 allow_empty=False，抛 GitError。
        """
        target = Path(worktree)
        self._ensure_committer_identity(target)
        if not self._has_staged_changes(target) and not allow_empty:
            raise GitError("没有可提交的变更")
        args = ["commit", "-m", message]
        if allow_empty:
            args.append("--allow-empty")
        self._run(args, cwd=target)
        return self.head_commit(target)

    def stage_and_commit(
        self,
        worktree: str | os.PathLike,
        message: str,
        *,
        allow_empty: bool = False,
    ) -> Optional[str]:
        """便捷方法：stage 全部 + commit。无变更且不允许空提交时返回 None。"""
        target = Path(worktree)
        self.stage_all(target)
        if not self._has_staged_changes(target) and not allow_empty:
            return None
        return self.commit(target, message, allow_empty=allow_empty)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def diff(
        self,
        worktree: str | os.PathLike,
        *,
        base: str,
        head: str = "HEAD",
    ) -> str:
        """unified diff 文本。"""
        result = self._run(["diff", f"{base}..{head}"], cwd=Path(worktree))
        return result.stdout

    def diff_stats(
        self,
        worktree: str | os.PathLike,
        *,
        base: str,
        head: str = "HEAD",
    ) -> dict:
        """`git diff --shortstat` 解析结果。"""
        result = self._run(
            ["diff", "--shortstat", f"{base}..{head}"],
            cwd=Path(worktree),
        )
        text = result.stdout.strip()
        stats = {"files": 0, "insertions": 0, "deletions": 0}
        if not text:
            return stats
        # 形如:" 3 files changed, 42 insertions(+), 5 deletions(-)"
        for token in text.split(","):
            token = token.strip()
            if "file" in token:
                stats["files"] = int(token.split()[0])
            elif "insertion" in token:
                stats["insertions"] = int(token.split()[0])
            elif "deletion" in token:
                stats["deletions"] = int(token.split()[0])
        return stats

    def changed_files(
        self,
        worktree: str | os.PathLike,
        *,
        base: str,
        head: str = "HEAD",
    ) -> List[str]:
        result = self._run(
            ["diff", "--name-only", f"{base}..{head}"],
            cwd=Path(worktree),
        )
        return [line for line in result.stdout.splitlines() if line.strip()]

    def show_commit(self, worktree: str | os.PathLike, sha: str) -> dict:
        """返回 sha 的元信息：作者、时间、消息。"""
        result = self._run(
            ["show", "-s", "--format=%H%n%an%n%aI%n%s%n%b", sha],
            cwd=Path(worktree),
        )
        lines = result.stdout.splitlines()
        return {
            "sha": lines[0] if len(lines) > 0 else sha,
            "author": lines[1] if len(lines) > 1 else "",
            "date": lines[2] if len(lines) > 2 else "",
            "subject": lines[3] if len(lines) > 3 else "",
            "body": "\n".join(lines[4:]) if len(lines) > 4 else "",
        }

    # ------------------------------------------------------------------
    # 回退
    # ------------------------------------------------------------------

    def reset_hard(self, worktree: str | os.PathLike, ref: str) -> None:
        """`git reset --hard <ref>`。慎用，仅在 worktree 内部使用。"""
        self._run(["reset", "--hard", ref], cwd=Path(worktree))


# ---------------------------------------------------------------------------
# 模块级单例（按需 import 即可）
# ---------------------------------------------------------------------------


_default_service: GitService | None = None


def get_git_service() -> GitService:
    """返回模块级 GitService 单例。测试中可显式新建实例。"""
    global _default_service
    if _default_service is None:
        _default_service = GitService()
    return _default_service
