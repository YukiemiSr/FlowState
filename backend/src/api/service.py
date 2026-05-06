"""Small service layer that keeps the API decoupled from runtime details."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from src.agents.base_agent import AgentInput
from src.agents.code_agent import CodeAgent
from src.agents.delivery_agent import DeliveryAgent
from src.agents.requirement_agent import RequirementAgent
from src.agents.review_agent import ReviewAgent
from src.agents.solution_agent import SolutionAgent
from src.agents.test_agent import TestAgent
from src.models.pipeline import (
    ApproveAction,
    GitContext,
    GitMode,
    Pipeline,
    PipelineContext,
    PipelineStatus,
    StageCommit,
    StageNode,
    StageStatus,
    StageType,
)
from src.services.git_service import GitError, GitService, get_git_service
from src.store.state_store import StateStore


def _build_default_stages() -> list[StageNode]:
    return [
        StageNode(stage_type=StageType.REQUIREMENT),
        StageNode(stage_type=StageType.SOLUTION),
        StageNode(stage_type=StageType.CODING),
        StageNode(stage_type=StageType.TESTING),
        StageNode(stage_type=StageType.REVIEW),
        StageNode(stage_type=StageType.DELIVERY),
    ]


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _running_stage(pipeline: Pipeline) -> StageNode | None:
    return next((stage for stage in pipeline.stages if stage.status == StageStatus.RUNNING), None)


def _apply_agent_metrics(stage: StageNode, output) -> None:
    usage = getattr(output, "token_usage", None) or {}
    if usage:
        stage.prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        stage.completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        stage.total_tokens = int(
            usage.get("total_tokens", 0)
            or stage.prompt_tokens
            or 0
        ) + (0 if usage.get("total_tokens", 0) else int(stage.completion_tokens or 0))
    if getattr(output, "model", None):
        stage.model_name = output.model
    if stage.agent_output is None:
        stage.agent_output = {}
    if usage:
        stage.agent_output["usage"] = {
            "prompt_tokens": stage.prompt_tokens or 0,
            "completion_tokens": stage.completion_tokens or 0,
            "total_tokens": stage.total_tokens or 0,
        }
    if stage.model_name:
        stage.agent_output["model"] = stage.model_name


def _append_usage_log(pipeline: Pipeline, stage: StageNode) -> None:
    if not stage.total_tokens:
        return
    pipeline.logs.append(
        f"[{_timestamp()}] {stage.stage_type.value} Token 消耗: "
        f"{stage.total_tokens}（prompt {stage.prompt_tokens or 0} / completion {stage.completion_tokens or 0}）"
    )


class PipelineValidationError(ValueError):
    """Raised when incoming pipeline creation data is invalid."""


STAGE_INDEX_MAP = {
    StageType.REQUIREMENT: 0,
    StageType.SOLUTION: 1,
    StageType.CODING: 2,
    StageType.TESTING: 3,
    StageType.REVIEW: 4,
    StageType.DELIVERY: 5,
}

STAGE_COMMIT_MESSAGE_MAP = {
    StageType.REQUIREMENT: "docs(flowstate): structured requirements",
    StageType.SOLUTION: "docs(flowstate): solution design",
    StageType.CODING: "feat(flowstate): implement {title}",
    StageType.TESTING: "test(flowstate): add tests and report",
    StageType.DELIVERY: "chore(flowstate): finalize delivery",
}

STAGE_DOC_FILENAMES = {
    StageType.REQUIREMENT: ["requirements.md"],
    StageType.SOLUTION: ["solution.md"],
    StageType.TESTING: ["test_report.md"],
    StageType.REVIEW: ["review_report.md"],
    StageType.DELIVERY: ["delivery.md", "pr.md"],
}


def _resolve_project_path(project_path: str) -> Path | None:
    if not project_path.strip():
        return None

    resolved_path = Path(project_path).expanduser().resolve()
    if not resolved_path.exists():
        raise PipelineValidationError(f"项目目录不存在: {resolved_path}")
    if not resolved_path.is_dir():
        raise PipelineValidationError(f"项目路径不是文件夹: {resolved_path}")
    return resolved_path


# ---------------------------------------------------------------------------
# Git 辅助：分支名生成
# ---------------------------------------------------------------------------

def _make_branch_name(pipeline_id: str, title: str) -> str:
    """生成形如 devflow/pipe_20260506_120000-my-feature 的分支名。"""
    slug = re.sub(r"[^\w一-鿿]+", "-", title.lower()).strip("-")
    # 只保留 ASCII 部分（中文 slug 对 git 合法但可读性差）
    slug_ascii = re.sub(r"[^\w-]+", "", slug)[:30].strip("-") or "task"
    return f"devflow/{pipeline_id}-{slug_ascii}"


# ---------------------------------------------------------------------------
# Git 辅助：pipeline 级 git 准备
# ---------------------------------------------------------------------------

def _prepare_git_for_pipeline(
    pipeline: Pipeline,
    git: GitService,
) -> None:
    """检测 / 初始化 git 仓库，创建 worktree，更新 pipeline.context.git。

    失败时不抛异常——降级为 DISABLED 模式，流水线继续正常运行。
    """
    project_path = pipeline.context.project_path
    if not project_path:
        return

    base_dir = Path(project_path)
    git_ctx = pipeline.context.git

    # ── git 可用性检查 ────────────────────────────────────────────────
    if not git.is_available():
        pipeline.logs.append(f"[{_timestamp()}] [Git] git CLI 不可用，跳过 Git 集成")
        return

    initialized = False
    repo_root: Path

    # ── 仓库状态探测 ──────────────────────────────────────────────────
    if git.is_git_repo(base_dir):
        repo_root = base_dir
    else:
        # 检查父目录是否是 git 仓库（嵌套仓库场景）
        enclosing = git.find_enclosing_repo(base_dir)
        if enclosing is not None:
            pipeline.logs.append(
                f"[{_timestamp()}] [Git] {base_dir} 位于上级仓库 {enclosing} 内部，"
                "不自动 init，跳过 Git 集成"
            )
            return
        # 自动 init
        try:
            git.init_repo(base_dir, default_branch="main")
            git.write_default_gitignore(base_dir)
            # 确保 .flowstate/ 在 .gitignore 里（worktree 路径）
            git.ensure_gitignore_entry(base_dir, ".flowstate/")
            git.baseline_commit(base_dir, message="chore: flowstate baseline")
            repo_root = base_dir
            initialized = True
            pipeline.logs.append(
                f"[{_timestamp()}] [Git] 已自动初始化 Git 仓库并创建 baseline commit"
            )
        except GitError as exc:
            pipeline.logs.append(
                f"[{_timestamp()}] [Git] 自动 init 失败: {exc}，跳过 Git 集成"
            )
            return

    # ── 创建 worktree ─────────────────────────────────────────────────
    base_commit = git.head_commit(repo_root) if git.has_any_commit(repo_root) else None
    if not base_commit:
        pipeline.logs.append(f"[{_timestamp()}] [Git] 仓库无任何 commit，跳过 Git 集成")
        return

    base_branch = git.current_branch(repo_root)
    branch_name = _make_branch_name(pipeline.id, pipeline.title)
    worktree_path = repo_root / ".flowstate" / "worktrees" / pipeline.id

    try:
        git.add_worktree(repo_root, worktree_path, branch=branch_name, base=base_commit)
        # worktree 内也确保 .gitignore 有 .flowstate/
        git.ensure_gitignore_entry(worktree_path, ".flowstate/")
    except GitError as exc:
        pipeline.logs.append(
            f"[{_timestamp()}] [Git] 创建 worktree 失败: {exc}，跳过 Git 集成"
        )
        return

    # ── 更新 GitContext ───────────────────────────────────────────────
    git_ctx.mode = GitMode.WORKTREE
    git_ctx.enabled = True
    git_ctx.repo_root = str(repo_root)
    git_ctx.base_branch = base_branch
    git_ctx.base_commit = base_commit
    git_ctx.worktree_path = str(worktree_path)
    git_ctx.working_branch = branch_name
    git_ctx.initialized = initialized

    pipeline.logs.append(
        f"[{_timestamp()}] [Git] Worktree 已就绪: {worktree_path}"
    )
    pipeline.logs.append(
        f"[{_timestamp()}] [Git] 工作分支: {branch_name}（基于 {base_branch}@{base_commit[:8]}）"
    )


def _effective_project_path(pipeline: Pipeline) -> str:
    """返回当前流水线写文件应用的目录。

    Git 启用时返回 worktree_path，否则返回原 project_path。
    """
    git_ctx = pipeline.context.git
    if git_ctx.enabled and git_ctx.worktree_path:
        return git_ctx.worktree_path
    return pipeline.context.project_path


def _effective_docs_path(pipeline: Pipeline) -> str:
    """返回流水线文档（requirements.md 等）应写入的目录。

    文档写到 <repo_root>/.flowstate/<pipeline_id>/docs/（或无 Git 时写到 project_path 下）。
    该目录由 .gitignore 忽略，不会被 commit 进 worktree。
    """
    git_ctx = pipeline.context.git
    if git_ctx.enabled and git_ctx.repo_root:
        base = git_ctx.repo_root
    elif pipeline.context.project_path:
        base = pipeline.context.project_path
    else:
        return ""
    docs_dir = Path(base) / ".flowstate" / pipeline.id / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    return str(docs_dir)


def _commit_stage(
    pipeline: Pipeline,
    stage_type: StageType,
    commit_message: str,
    git: GitService,
) -> None:
    """在 worktree 中提交当前阶段产物，更新 GitContext。

    若 Git 未启用或提交失败，静默跳过（不影响流水线）。
    """
    git_ctx = pipeline.context.git
    if not git_ctx.enabled or not git_ctx.worktree_path:
        return

    wt = git_ctx.worktree_path
    try:
        sha = git.stage_and_commit(wt, commit_message)
        if sha is None:
            # 无变更——可能 stage 没写任何文件，记录一下
            pipeline.logs.append(
                f"[{_timestamp()}] [Git] {stage_type.value}: 无文件变更，跳过 commit"
            )
            return

        now = datetime.now()
        changed = git.changed_files(wt, base=git_ctx.base_commit or "HEAD~1", head=sha)
        stage_commit = StageCommit(
            stage_type=stage_type,
            commit_sha=sha,
            commit_message=commit_message,
            committed_at=now,
            files_changed=changed,
        )
        git_ctx.stage_commits.append(stage_commit)
        git_ctx.head_commit = sha
        for f in changed:
            if f not in git_ctx.total_files_changed:
                git_ctx.total_files_changed.append(f)

        pipeline.logs.append(
            f"[{_timestamp()}] [Git] commit {sha[:8]}: {commit_message}"
        )
    except GitError as exc:
        pipeline.logs.append(
            f"[{_timestamp()}] [Git] commit 失败（{stage_type.value}）: {exc}"
        )


def _summarize_project_path(project_path: Path) -> str:
    top_level_dirs: list[str] = []
    top_level_files: list[str] = []
    for item in sorted(project_path.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower())):
        if item.is_dir():
            top_level_dirs.append(item.name)
        else:
            top_level_files.append(item.name)

    total_dirs = 0
    total_files = 0
    for _, dirnames, filenames in os.walk(project_path):
        total_dirs += len(dirnames)
        total_files += len(filenames)

    key_files = [
        name for name in (
            "package.json",
            "pnpm-lock.yaml",
            "package-lock.json",
            "yarn.lock",
            "pyproject.toml",
            "requirements.txt",
            "Cargo.toml",
            "go.mod",
            "Dockerfile",
            "README.md",
        )
        if (project_path / name).exists()
    ]

    preview_dirs = "、".join(top_level_dirs[:6]) if top_level_dirs else "无"
    preview_files = "、".join(top_level_files[:6]) if top_level_files else "无"
    preview_key_files = "、".join(key_files) if key_files else "未识别到常见入口文件"

    return (
        "## 项目目录扫描\n\n"
        f"- 根目录：{project_path}\n"
        f"- 顶层目录：{preview_dirs}\n"
        f"- 顶层文件：{preview_files}\n"
        f"- 关键文件：{preview_key_files}\n"
        f"- 目录总数：{total_dirs}\n"
        f"- 文件总数：{total_files}"
    )


def _write_project_doc(docs_path: str, filename: str, content: str) -> Path:
    """将阶段文档写到 docs_path 目录下。

    docs_path 由 _effective_docs_path(pipeline) 提供，
    通常是 <worktree>/.flowstate/<pipeline_id>/docs/。
    """
    if not docs_path:
        raise PipelineValidationError("缺少文档目录，无法写入阶段文档")

    docs_dir = Path(docs_path)
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_path = docs_dir / filename
    doc_path.write_text(content, encoding="utf-8")
    return doc_path


def _normalize_generated_filepath(filepath: str) -> Path:
    cleaned = filepath.strip().lstrip("/").replace("\\", "/")
    relative_path = Path(cleaned)
    if not cleaned or any(part == ".." for part in relative_path.parts):
        raise PipelineValidationError(f"生成了非法文件路径: {filepath}")
    return relative_path


def _write_generated_code(project_paths: list[str], files: dict[str, str]) -> list[str]:
    if not project_paths:
        raise PipelineValidationError("缺少项目目录，无法写入生成代码")

    normalized_bases: list[Path] = []
    seen: set[str] = set()
    for project_path in project_paths:
        normalized = str(Path(project_path))
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_bases.append(Path(project_path))

    preferred_base = normalized_bases[-1]
    written_files: list[str] = []
    for filepath, content in files.items():
        relative_path = _normalize_generated_filepath(filepath)
        preferred_target = preferred_base / relative_path
        for base_dir in normalized_bases:
            target_path = base_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
        written_files.append(str(preferred_target))
    return written_files


def _remove_root_relative_paths(root_path: str, relative_paths: set[str]) -> None:
    if not root_path:
        return

    root = Path(root_path)
    for relative_path in sorted(relative_paths):
        target = root / relative_path
        if target.is_file():
            target.unlink()


def _remove_future_stage_docs(pipeline: Pipeline, stage_index: int) -> None:
    root_path = _root_project_path(pipeline)
    if not root_path:
        return

    root = Path(root_path)
    for stage in pipeline.stages[stage_index:]:
        for filename in STAGE_DOC_FILENAMES.get(stage.stage_type, []):
            target = _resolve_doc_dir(root, pipeline) / filename
            if target.exists():
                target.unlink()


class PipelineService:
    """A tiny adapter around either an injected engine or local persistence."""

    def __init__(
        self,
        engine=None,
        state_store: StateStore | None = None,
        git_service: GitService | None = None,
    ):
        self.engine = engine
        self.state_store = state_store or getattr(engine, "state_store", None) or StateStore()
        self.git = git_service or get_git_service()

    async def create_pipeline(
        self,
        *,
        title: str,
        requirement: str,
        project_path: str = "",
        start_immediately: bool = False,
    ) -> Pipeline:
        resolved_project_path = _resolve_project_path(project_path)
        normalized_project_path = str(resolved_project_path) if resolved_project_path else project_path.strip()
        project_summary = (
            _summarize_project_path(resolved_project_path)
            if resolved_project_path is not None
            else None
        )

        if self.engine is not None:
            pipeline = await self.engine.create_pipeline(requirement=requirement, title=title)
            pipeline.context.project_path = normalized_project_path
            pipeline.context.project_summary = project_summary
        else:
            now = datetime.now()
            pipeline = Pipeline(
                title=title or requirement[:60],
                status=PipelineStatus.PENDING,
                context=PipelineContext(
                    project_path=normalized_project_path,
                    project_summary=project_summary,
                    requirement_raw=requirement,
                ),
                stages=_build_default_stages(),
                created_at=now,
                updated_at=now,
            )

        if not pipeline.logs:
            pipeline.logs = [
                f"[{_timestamp()}] Pipeline 已创建",
                *([f"[{_timestamp()}] 工作目录: {normalized_project_path}"] if normalized_project_path else []),
                *([f"[{_timestamp()}] 已接收需求: {requirement[:80]}"] if requirement else []),
                *([f"[{_timestamp()}] 项目目录扫描完成"] if project_summary else []),
            ]

        # ── Git 准备（非阻塞：失败则降级为 DISABLED 模式）────────────────
        if normalized_project_path:
            _prepare_git_for_pipeline(pipeline, self.git)

        if start_immediately and pipeline.stages:
            now = datetime.now()
            pipeline.status = PipelineStatus.RUNNING
            pipeline.updated_at = now
            pipeline.stages[0].status = StageStatus.RUNNING
            pipeline.stages[0].started_at = now
            pipeline.stages[0].agent_output = {
                "text": (
                    "## 已接收任务\n\n"
                    f"项目目录：{normalized_project_path or '未提供'}\n\n"
                    f"需求描述：{requirement}\n\n"
                    f"{project_summary or '未执行项目目录扫描。'}"
                )
            }
            if not any("RequirementsAgent 正在分析需求" in item for item in pipeline.logs):
                pipeline.logs.append(f"[{_timestamp()}] RequirementsAgent 正在分析需求...")

        if start_immediately and pipeline.stages:
            await self._run_requirement_analysis(pipeline)

        await self.state_store.save(pipeline)
        return pipeline

    async def get_pipeline(self, pipeline_id: str) -> Pipeline | None:
        return await self.state_store.load(pipeline_id)

    async def list_pipelines(self) -> list[Pipeline]:
        return await self.state_store.list_pipelines()

    async def delete_pipeline(self, pipeline_id: str) -> bool:
        pipeline = await self.state_store.load(pipeline_id)
        if pipeline is None:
            return False
        self._cleanup_git_context(pipeline)
        await self.state_store.delete(pipeline_id)
        return True

    async def cleanup_pipeline_git(self, pipeline_id: str) -> Pipeline:
        pipeline = await self.state_store.load(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        self._cleanup_git_context(pipeline)
        await self.state_store.save(pipeline)
        return pipeline

    def git_diff_for_pipeline(self, pipeline: Pipeline, stage_type: StageType | None = None) -> str:
        git_ctx = pipeline.context.git
        if not git_ctx.enabled or not git_ctx.worktree_path or not git_ctx.base_commit:
            return ""

        worktree = Path(git_ctx.worktree_path)
        if stage_type is None:
            head = git_ctx.head_commit or "HEAD"
            return self.git_service.diff(worktree, base=git_ctx.base_commit, head=head)

        stage_commit = next(
            (item for item in git_ctx.stage_commits if item.stage_type == stage_type),
            None,
        )
        if stage_commit is None:
            return ""

        prev_candidates = [
            item
            for item in git_ctx.stage_commits
            if STAGE_INDEX_MAP[item.stage_type] < STAGE_INDEX_MAP[stage_type]
        ]
        base = prev_candidates[-1].commit_sha if prev_candidates else git_ctx.base_commit
        return self.git_service.diff(worktree, base=base, head=stage_commit.commit_sha)

    def _prepare_git_context(self, *, pipeline: Pipeline, resolved_project_path: Path | None) -> None:
        cfg = get_config().git
        if not cfg.enabled or resolved_project_path is None:
            pipeline.context.git = GitContext(mode=GitMode.DISABLED, enabled=False)
            return

        target_path = resolved_project_path
        initialized = False
        repo_root = self.git_service.find_repo_root(target_path)
        if repo_root is not None and repo_root != target_path:
            raise NestedRepoError(
                f"路径 {target_path} 位于仓库 {repo_root} 内，请选择仓库根目录"
            )
        if repo_root is None:
            enclosing_repo = self.git_service.find_enclosing_repo(target_path)
            if enclosing_repo is not None and enclosing_repo != target_path:
                raise NestedRepoError(
                    f"路径 {target_path} 位于仓库 {enclosing_repo} 内，请选择仓库根目录"
                )
            if not cfg.auto_init_if_missing:
                pipeline.context.git = GitContext(mode=GitMode.DISABLED, enabled=False)
                return
            self.git_service.write_default_gitignore(target_path)
            self.git_service.init_repo(target_path, default_branch=cfg.default_base_branch)
            self.git_service.baseline_commit(target_path)
            initialized = True
            repo_root = target_path
        elif self.git_service.has_changes(repo_root):
            raise GitError(f"Git 仓库存在未提交变更，请先提交或清理工作区: {repo_root}")

        base_branch = self.git_service.current_branch(repo_root)
        base_commit = self.git_service.head_commit(repo_root)
        working_branch = self._build_working_branch_name(pipeline)
        worktree_path = repo_root / ".flowstate" / "worktrees" / pipeline.id
        try:
            self.git_service.delete_branch(repo_root, working_branch, force=True)
        except GitError:
            pass
        self.git_service.add_worktree(
            repo=repo_root,
            worktree_path=worktree_path,
            branch=working_branch,
            base=base_commit,
        )
        self.git_service.ensure_flowstate_worktree_ignore(repo_root)

        pipeline.context.git = GitContext(
            mode=GitMode.WORKTREE,
            enabled=True,
            repo_root=str(repo_root),
            base_branch=base_branch,
            base_commit=base_commit,
            worktree_path=str(worktree_path),
            working_branch=working_branch,
            initialized=initialized,
            stage_commits=[],
            total_files_changed=[],
            head_commit=self.git_service.head_commit(worktree_path),
            diff_stats={"files": 0, "insertions": 0, "deletions": 0},
        )
        pipeline.logs.append(
            f"[{_timestamp()}] 已创建工作分支 {working_branch}，工作目录 {worktree_path}"
        )

    def _build_working_branch_name(self, pipeline: Pipeline) -> str:
        cfg = get_config().git
        slug = self._slugify_title(pipeline.title or pipeline.context.requirement_raw)
        template = cfg.branch_naming_template or "devflow/{pipeline_id}-{slug}"
        branch = template.replace("{pipeline_id}", pipeline.id).replace("{slug}", slug)
        branch = re.sub(r"/+", "/", branch).strip("/")
        return branch or f"devflow/{pipeline.id}"

    def _slugify_title(self, text: str) -> str:
        lowered = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
        if not lowered:
            return "task"
        return lowered[:24]

    def _build_agent_input_context(self, pipeline: Pipeline) -> dict:
        context = pipeline.context.model_dump()
        context["project_path"] = _root_project_path(pipeline)
        context["execution_path"] = _execution_project_path(pipeline)
        return context

    def _commit_stage(self, pipeline: Pipeline, stage_type: StageType, commit_message: str) -> None:
        cfg = get_config().git
        git_ctx = pipeline.context.git
        if not cfg.commit_per_stage:
            return
        if stage_type == StageType.REVIEW and not cfg.commit_review_report:
            return
        if not git_ctx.enabled or not git_ctx.worktree_path:
            return

        worktree = Path(git_ctx.worktree_path)
        if not self.git_service.has_changes(worktree):
            return

        self.git_service.stage_all(worktree)
        sha = self.git_service.commit(worktree, commit_message)
        prev_commit = self._find_prev_commit_anchor(pipeline, STAGE_INDEX_MAP.get(stage_type, 999))
        stage_base = prev_commit.commit_sha if prev_commit is not None else (git_ctx.base_commit or sha)
        files_changed = self.git_service.changed_files(
            worktree,
            base=stage_base,
            head=sha,
        )
        git_ctx.stage_commits.append(
            StageCommit(
                stage_type=stage_type,
                commit_sha=sha,
                commit_message=commit_message,
                committed_at=datetime.now(),
                files_changed=files_changed,
            )
        )
        git_ctx.head_commit = sha
        if git_ctx.base_commit:
            git_ctx.diff_stats = self.git_service.diff_stats(worktree, base=git_ctx.base_commit, head=sha)
            git_ctx.total_files_changed = self.git_service.changed_files(
                worktree,
                base=git_ctx.base_commit,
                head=sha,
            )
        pipeline.logs.append(f"[{_timestamp()}] [{stage_type.value}] commit {sha[:7]}: {commit_message}")

    def _default_commit_message(self, pipeline: Pipeline, stage_type: StageType) -> str:
        template = STAGE_COMMIT_MESSAGE_MAP.get(stage_type)
        if template is None:
            return f"chore(flowstate): update {stage_type.value}"
        short_title = self._slugify_title(pipeline.title or pipeline.context.requirement_raw).replace("-", " ")
        return template.format(title=short_title or "changes")

    def _find_prev_commit_anchor(self, pipeline: Pipeline, stage_index: int) -> StageCommit | None:
        candidates = [
            commit
            for commit in pipeline.context.git.stage_commits
            if STAGE_INDEX_MAP.get(commit.stage_type, 999) < stage_index
        ]
        return candidates[-1] if candidates else None

    def _reset_git_to_anchor(self, pipeline: Pipeline, stage_index: int) -> None:
        git_ctx = pipeline.context.git
        if not git_ctx.enabled or not git_ctx.worktree_path:
            return
        target_commit = self._find_prev_commit_anchor(pipeline, stage_index)
        target_sha = target_commit.commit_sha if target_commit else git_ctx.base_commit
        if not target_sha:
            return

        try:
            previous_total_files = set(git_ctx.total_files_changed)
            worktree = Path(git_ctx.worktree_path)
            self.git_service.reset_hard(worktree, target_sha)
            self.git_service.clean_untracked(worktree)
            git_ctx.stage_commits = [
                commit
                for commit in git_ctx.stage_commits
                if STAGE_INDEX_MAP.get(commit.stage_type, 999) < stage_index
            ]
            git_ctx.head_commit = target_sha
            if git_ctx.base_commit:
                git_ctx.diff_stats = self.git_service.diff_stats(
                    worktree,
                    base=git_ctx.base_commit,
                    head=target_sha,
                )
                git_ctx.total_files_changed = self.git_service.changed_files(
                    worktree,
                    base=git_ctx.base_commit,
                    head=target_sha,
                )
            _remove_root_relative_paths(
                _root_project_path(pipeline),
                previous_total_files - set(git_ctx.total_files_changed),
            )
            _remove_future_stage_docs(pipeline, stage_index)
            pipeline.logs.append(f"[{_timestamp()}] Git 已回退到 {target_sha[:7]}")
        except GitError as error:
            pipeline.logs.append(f"[{_timestamp()}] Git 回退失败: {error}")

    def _cleanup_git_context(self, pipeline: Pipeline) -> None:
        git_ctx = pipeline.context.git
        if not git_ctx.enabled:
            return

        repo_root = Path(git_ctx.repo_root) if git_ctx.repo_root else None
        worktree_path = Path(git_ctx.worktree_path) if git_ctx.worktree_path else None
        worktree_removed = worktree_path is None
        branch_removed = not bool(git_ctx.working_branch)

        if repo_root is not None and worktree_path is not None:
            try:
                self.git_service.remove_worktree(repo_root, worktree_path, force=True)
                worktree_removed = True
            except GitError as error:
                pipeline.logs.append(f"[{_timestamp()}] 清理 worktree 失败: {error}")

        if repo_root is not None and git_ctx.working_branch:
            try:
                self.git_service.delete_branch(repo_root, git_ctx.working_branch, force=True)
                branch_removed = True
            except GitError as error:
                pipeline.logs.append(f"[{_timestamp()}] 删除分支失败: {error}")

        if worktree_removed:
            git_ctx.worktree_path = None
        if branch_removed:
            git_ctx.working_branch = None

        if worktree_removed and branch_removed:
            git_ctx.mode = GitMode.DISABLED
            git_ctx.enabled = False
            git_ctx.head_commit = git_ctx.base_commit
        else:
            pipeline.logs.append(f"[{_timestamp()}] Git 清理未完成，保留当前 Git 上下文以便重试")

    async def pause_pipeline(self, pipeline_id: str) -> Pipeline:
        pipeline = await self.state_store.load(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        if pipeline.status != PipelineStatus.RUNNING:
            raise PipelineValidationError("只有运行中的流水线才能暂停")

        active_stage = _running_stage(pipeline)
        if active_stage is not None:
            active_stage.status = StageStatus.PENDING
        pipeline.status = PipelineStatus.PAUSED
        pipeline.updated_at = datetime.now()
        pipeline.logs.append(f"[{_timestamp()}] 人工暂停流水线")
        await self.state_store.save(pipeline)
        return pipeline

    async def resume_pipeline(self, pipeline_id: str) -> Pipeline:
        pipeline = await self.state_store.load(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        if pipeline.status == PipelineStatus.WAITING_HUMAN:
            raise PipelineValidationError("当前流水线在人工审批检查点暂停，请前往检查点页处理")
        if pipeline.status != PipelineStatus.PAUSED:
            raise PipelineValidationError("只有已暂停的流水线才能继续")

        next_stage = next(
            (stage for stage in pipeline.stages if stage.status == StageStatus.PENDING),
            None,
        )
        if next_stage is not None:
            next_stage.status = StageStatus.RUNNING
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = datetime.now()
        pipeline.logs.append(f"[{_timestamp()}] 继续执行流水线")
        await self.state_store.save(pipeline)
        return pipeline

    async def cancel_pipeline(self, pipeline_id: str) -> Pipeline:
        pipeline = await self.state_store.load(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        if pipeline.status in {PipelineStatus.COMPLETED, PipelineStatus.CANCELLED}:
            raise PipelineValidationError("当前流水线无法终止")

        active_stage = _running_stage(pipeline)
        if active_stage is not None:
            active_stage.status = StageStatus.PENDING
        pipeline.status = PipelineStatus.CANCELLED
        pipeline.updated_at = datetime.now()
        pipeline.logs.append(f"[{_timestamp()}] 人工终止流水线")
        await self.state_store.save(pipeline)
        return pipeline

    async def retry_pipeline(self, pipeline_id: str) -> Pipeline:
        pipeline = await self.state_store.load(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        if pipeline.status != PipelineStatus.FAILED:
            raise PipelineValidationError("只有失败的流水线才能重试")

        failed_index = next(
            (index for index, stage in enumerate(pipeline.stages) if stage.status == StageStatus.FAILED),
            None,
        )
        if failed_index is None:
            raise PipelineValidationError("未找到可重试的失败阶段")

        for stage in pipeline.stages[failed_index:]:
            stage.status = StageStatus.PENDING
            stage.agent_output = None
            stage.prompt_tokens = None
            stage.completion_tokens = None
            stage.total_tokens = None
            stage.model_name = None
            stage.human_feedback = None
            stage.human_approval = None
            stage.started_at = None
            stage.completed_at = None
            stage.retry_count += 1

        self._reset_git_to_anchor(pipeline, failed_index)
        pipeline.status = PipelineStatus.RUNNING
        pipeline.error = None
        pipeline.updated_at = datetime.now()
        pipeline.logs.append(
            f"[{_timestamp()}] 人工触发重试，从阶段 {pipeline.stages[failed_index].stage_type.value} 重新开始"
        )
        await self.state_store.save(pipeline)
        await self._run_stage_by_index(pipeline, failed_index)
        return pipeline

    async def run_pipeline_by_id(self, pipeline_id: str) -> Pipeline:
        pipeline = await self.state_store.load(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        return await self.run_pipeline(pipeline)

    async def run_pipeline(self, pipeline: Pipeline) -> Pipeline:
        if self.engine is not None:
            await self.engine.run_pipeline(pipeline)
            return pipeline

        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = datetime.now()
        await self.state_store.save(pipeline)
        return pipeline

    async def continue_after_approval(self, pipeline_id: str, approved_stage_index: int) -> Pipeline | None:
        pipeline = await self.state_store.load(pipeline_id)
        if pipeline is None:
            return None

        next_stage_index = approved_stage_index + 1
        if next_stage_index >= len(pipeline.stages):
            await self.state_store.save(pipeline)
            return pipeline

        await self._run_stage_by_index(pipeline, next_stage_index)
        return pipeline

    async def approve_stage(self, pipeline_id: str, stage_index: int) -> Pipeline:
        pipeline = await self.state_store.load(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        if stage_index < 0 or stage_index >= len(pipeline.stages):
            raise ValueError(f"Invalid stage index: {stage_index}")

        stage = pipeline.stages[stage_index]
        if stage.status != StageStatus.WAITING_HUMAN:
            raise ValueError("当前阶段不处于待审批状态")

        now = datetime.now()
        stage.human_approval = ApproveAction.APPROVE
        stage.status = StageStatus.APPROVED
        stage.completed_at = stage.completed_at or now
        pipeline.updated_at = now
        pipeline.logs.append(f"[{_timestamp()}] 人工审批通过: {stage.stage_type.value}")

        if stage.stage_type == StageType.DELIVERY:
            pipeline.status = PipelineStatus.COMPLETED
            pipeline.logs.append(f"[{_timestamp()}] 所有阶段已完成")
            await self.state_store.save(pipeline)
            return pipeline

        next_stage = pipeline.stages[stage_index + 1] if stage_index + 1 < len(pipeline.stages) else None
        if next_stage is not None:
            next_stage.status = StageStatus.RUNNING
            next_stage.started_at = now
            if not next_stage.agent_output:
                next_stage.agent_output = {
                    "text": (
                        f"## {next_stage.stage_type.value} 已进入执行队列\n\n"
                        "上一阶段审批已通过，当前阶段已进入后台执行。"
                    )
                }
            pipeline.status = PipelineStatus.RUNNING
            pipeline.logs.append(f"[{_timestamp()}] 已推进到下一阶段: {next_stage.stage_type.value}")
            pipeline.logs.append(f"[{_timestamp()}] 已提交后台执行任务")
        else:
            pipeline.status = PipelineStatus.COMPLETED
            pipeline.logs.append(f"[{_timestamp()}] 所有阶段已完成")

        await self.state_store.save(pipeline)
        return pipeline

    async def reject_stage(self, pipeline_id: str, stage_index: int, reason: str) -> Pipeline:
        pipeline = await self.state_store.load(pipeline_id)
        if pipeline is None:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        if stage_index < 0 or stage_index >= len(pipeline.stages):
            raise ValueError(f"Invalid stage index: {stage_index}")

        stage = pipeline.stages[stage_index]
        if stage.status != StageStatus.WAITING_HUMAN:
            raise ValueError("当前阶段不处于待审批状态")

        now = datetime.now()
        stage.human_approval = ApproveAction.REJECT
        stage.human_feedback = reason
        stage.agent_output = None
        stage.prompt_tokens = None
        stage.completion_tokens = None
        stage.total_tokens = None
        stage.model_name = None
        stage.status = StageStatus.REJECTED
        stage.started_at = None
        stage.completed_at = None
        stage.retry_count += 1
        pipeline.updated_at = now
        pipeline.status = PipelineStatus.RUNNING
        pipeline.logs.append(f"[{_timestamp()}] 人工审批拒绝: {stage.stage_type.value}")
        if reason:
            pipeline.logs.append(f"[{_timestamp()}] 拒绝原因: {reason}")
        pipeline.logs.append(
            f"[{_timestamp()}] Stage 状态重置为 PENDING，并提交后台 Agent 重新执行..."
        )

        # ── Git：回退 worktree 到上一个 stage 的 commit 锚点 ──────────
        git_ctx = pipeline.context.git
        if git_ctx.enabled and git_ctx.worktree_path and git_ctx.stage_commits:
            # 找到当前被拒绝 stage 之前最近的一个 commit 锚点
            reset_target: str | None = None
            for sc in reversed(git_ctx.stage_commits):
                if sc.stage_type != stage.stage_type:
                    reset_target = sc.commit_sha
                    break
            if reset_target is None:
                reset_target = git_ctx.base_commit  # 回到起点
            if reset_target:
                try:
                    self.git.reset_hard(git_ctx.worktree_path, reset_target)
                    git_ctx.head_commit = reset_target
                    # 移除被 reject 的 stage commit 记录
                    git_ctx.stage_commits = [
                        sc for sc in git_ctx.stage_commits
                        if sc.stage_type != stage.stage_type
                    ]
                    pipeline.logs.append(
                        f"[{_timestamp()}] [Git] Worktree 已回退到 {reset_target[:8]}"
                    )
                except GitError as exc:
                    pipeline.logs.append(
                        f"[{_timestamp()}] [Git] Worktree 回退失败: {exc}"
                    )

        await self.state_store.save(pipeline)
        return pipeline

    async def retry_stage(self, pipeline_id: str, stage_index: int) -> None:
        """后台重新执行指定阶段（由 reject/retry 在后台调用）。"""
        # 注意：此处不需要再次设置 PENDING，因为 reject_stage 已经完成了状态初步重置
        # 我们确保进入运行状态并启动
        pipeline = await self.state_store.load(pipeline_id)
        if pipeline is None:
            return
        if stage_index < 0 or stage_index >= len(pipeline.stages):
            return
        stage = pipeline.stages[stage_index]
        stage.status = StageStatus.RUNNING
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = datetime.now()
        pipeline.logs.append(
            f"[{_timestamp()}] 正在重新执行阶段 {stage.stage_type.value}"
        )
        await self.state_store.save(pipeline)
        try:
            await self._run_stage_by_index(pipeline, stage_index)
        except Exception:
            pass

    async def _run_requirement_analysis(self, pipeline: Pipeline) -> None:
        """执行需求分析阶段，生成第一版结构化需求文档。"""
        if not pipeline.stages:
            return

        requirement_stage = pipeline.stages[0]
        now = datetime.now()
        requirement_stage.status = StageStatus.RUNNING
        requirement_stage.started_at = now
        requirement_stage.agent_output = {
            "text": (
                "## 正在分析需求\n\n"
                "已接收任务，正在根据需求描述和项目上下文进行分析。"
            )
        }
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = now
        pipeline.logs.append(f"[{_timestamp()}] RequirementsAgent 正在分析需求...")

        try:
            if self.engine is not None and StageType.REQUIREMENT in getattr(self.engine, "agents", {}):
                agent = self.engine.agents[StageType.REQUIREMENT]
            else:
                agent = RequirementAgent()

            input_data = AgentInput(
                task_description="执行阶段: requirement_analysis",
                context=self._build_agent_input_context(pipeline),
                human_feedback=requirement_stage.human_feedback,
            )
            output = await agent.execute(input_data)
            now = datetime.now()
            requirement_stage.agent_output = output.result
            _apply_agent_metrics(requirement_stage, output)
            requirement_stage.status = (
                StageStatus.WAITING_HUMAN
                if output.needs_human_review
                else StageStatus.COMPLETED
            )
            requirement_stage.completed_at = now
            requirement_doc = output.result.get("document") or output.details
            pipeline.context.requirement_doc = requirement_doc
            if requirement_doc:
                requirement_doc_path = _write_project_doc(
                    _effective_docs_path(pipeline),
                    "requirements.md",
                    requirement_doc,
                )
                pipeline.context.requirement_doc_path = str(requirement_doc_path)
            pipeline.updated_at = now
            pipeline.status = (
                PipelineStatus.WAITING_HUMAN
                if output.needs_human_review
                else PipelineStatus.PENDING
            )
            pipeline.logs.append(f"[{_timestamp()}] {output.summary}")
            _append_usage_log(pipeline, requirement_stage)
            if pipeline.context.requirement_doc_path:
                pipeline.logs.append(
                    f"[{_timestamp()}] 需求文档已写入: {pipeline.context.requirement_doc_path}"
                )
            self._commit_stage(
                pipeline,
                StageType.REQUIREMENT,
                self._default_commit_message(pipeline, StageType.REQUIREMENT),
            )
            if output.needs_human_review:
                pipeline.logs.append(f"[{_timestamp()}] 需求分析进入人工确认，等待审批")
            await self.state_store.save(pipeline)
        except Exception as error:
            requirement_stage.status = StageStatus.FAILED
            requirement_stage.completed_at = datetime.now()
            pipeline.status = PipelineStatus.FAILED
            pipeline.error = f"[requirement_analysis] {error}"
            pipeline.logs.append(f"[{_timestamp()}] 需求分析失败: {error}")
            await self.state_store.save(pipeline)
            raise

    async def _run_stage_by_index(self, pipeline: Pipeline, stage_index: int) -> None:
        stage_type = pipeline.stages[stage_index].stage_type
        if stage_type == StageType.REQUIREMENT:
            await self._run_requirement_analysis(pipeline)
        elif stage_type == StageType.SOLUTION:
            await self._run_solution_design(pipeline)
        elif stage_type == StageType.CODING:
            await self._run_code_generation(pipeline)
        elif stage_type == StageType.TESTING:
            await self._run_testing(pipeline)
        elif stage_type == StageType.REVIEW:
            await self._run_review(pipeline)
        elif stage_type == StageType.DELIVERY:
            await self._run_delivery(pipeline)

    async def _run_solution_design(self, pipeline: Pipeline) -> None:
        if len(pipeline.stages) < 2:
            return

        solution_stage = pipeline.stages[1]
        now = datetime.now()
        solution_stage.status = StageStatus.RUNNING
        solution_stage.started_at = now
        solution_stage.agent_output = {
            "text": (
                "## 正在生成技术方案\n\n"
                "需求分析已审批通过，正在根据需求文档和项目上下文生成方案设计。"
            )
        }
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = now
        pipeline.logs.append(f"[{_timestamp()}] SolutionAgent 正在生成技术方案...")

        try:
            if self.engine is not None and StageType.SOLUTION in getattr(self.engine, "agents", {}):
                agent = self.engine.agents[StageType.SOLUTION]
            else:
                agent = SolutionAgent()

            input_data = AgentInput(
                task_description="执行阶段: solution_design",
                context=self._build_agent_input_context(pipeline),
                human_feedback=solution_stage.human_feedback,
            )
            output = await agent.execute(input_data)
            now = datetime.now()
            solution_stage.agent_output = output.result
            _apply_agent_metrics(solution_stage, output)
            solution_stage.status = (
                StageStatus.WAITING_HUMAN
                if output.needs_human_review
                else StageStatus.COMPLETED
            )
            solution_stage.completed_at = now
            solution_doc = output.result.get("design") or output.details
            pipeline.context.solution_doc = solution_doc
            structured_solution = output.result.get("structured_solution")
            if isinstance(structured_solution, dict):
                pipeline.context.solution_structured = structured_solution
            if solution_doc:
                solution_doc_path = _write_project_doc(
                    _effective_docs_path(pipeline),
                    "solution.md",
                    solution_doc,
                )
                pipeline.context.solution_doc_path = str(solution_doc_path)
            pipeline.updated_at = now
            pipeline.status = (
                PipelineStatus.WAITING_HUMAN
                if output.needs_human_review
                else PipelineStatus.PENDING
            )
            pipeline.logs.append(f"[{_timestamp()}] {output.summary}")
            _append_usage_log(pipeline, solution_stage)
            if pipeline.context.solution_doc_path:
                pipeline.logs.append(
                    f"[{_timestamp()}] 技术方案文档已写入: {pipeline.context.solution_doc_path}"
                )
            self._commit_stage(
                pipeline,
                StageType.SOLUTION,
                self._default_commit_message(pipeline, StageType.SOLUTION),
            )
            if output.needs_human_review:
                pipeline.logs.append(f"[{_timestamp()}] 方案设计进入人工确认，等待审批")
            await self.state_store.save(pipeline)
        except Exception as error:
            solution_stage.status = StageStatus.FAILED
            solution_stage.completed_at = datetime.now()
            pipeline.status = PipelineStatus.FAILED
            pipeline.error = f"[solution_design] {error}"
            pipeline.logs.append(f"[{_timestamp()}] 方案设计失败: {error}")
            await self.state_store.save(pipeline)
            raise

    async def _run_code_generation(self, pipeline: Pipeline) -> None:
        if len(pipeline.stages) < 3:
            return

        coding_stage = pipeline.stages[2]
        now = datetime.now()
        coding_stage.status = StageStatus.RUNNING
        coding_stage.started_at = now
        coding_stage.agent_output = {
            "text": (
                "## 正在生成代码\n\n"
                "方案设计已审批通过，正在根据技术方案生成项目代码。"
            )
        }
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = now
        pipeline.logs.append(f"[{_timestamp()}] CodeAgent 正在生成代码...")

        try:
            if self.engine is not None and StageType.CODING in getattr(self.engine, "agents", {}):
                agent = self.engine.agents[StageType.CODING]
            else:
                agent = CodeAgent()

            input_data = AgentInput(
                task_description="执行阶段: coding",
                context=self._build_agent_input_context(pipeline),
                human_feedback=coding_stage.human_feedback,
            )
            output = await agent.execute(input_data)
            now = datetime.now()
            coding_stage.agent_output = output.result
            _apply_agent_metrics(coding_stage, output)
            coding_stage.status = StageStatus.COMPLETED
            coding_stage.completed_at = now

            generated_files = output.result.get("files")
            if isinstance(generated_files, dict):
                pipeline.context.generated_code = generated_files
                written_files = _write_generated_code(
                    _effective_project_path(pipeline),
                    generated_files,
                )
            else:
                written_files = []

            pipeline.updated_at = now
            pipeline.status = PipelineStatus.PENDING
            pipeline.logs.append(f"[{_timestamp()}] {output.summary}")
            _append_usage_log(pipeline, coding_stage)
            if written_files:
                preview = "、".join(Path(path).name for path in written_files[:5])
                pipeline.logs.append(f"[{_timestamp()}] 生成代码已写入: {preview}")
            _commit_stage(
                pipeline, StageType.CODING,
                "feat: generate code via FlowState (stage3)",
                self.git,
            )
            await self.state_store.save(pipeline)
            await self._run_testing(pipeline)
        except Exception as error:
            coding_stage.status = StageStatus.FAILED
            coding_stage.completed_at = datetime.now()
            pipeline.status = PipelineStatus.FAILED
            pipeline.error = f"[coding] {error}"
            pipeline.logs.append(f"[{_timestamp()}] 代码生成失败: {error}")
            await self.state_store.save(pipeline)
            raise

    async def _run_testing(self, pipeline: Pipeline) -> None:
        if len(pipeline.stages) < 4:
            return

        testing_stage = pipeline.stages[3]
        now = datetime.now()
        testing_stage.status = StageStatus.RUNNING
        testing_stage.started_at = now
        testing_stage.agent_output = {
            "text": (
                "## 正在生成与整理测试结果\n\n"
                "代码生成已完成，正在根据变更文件生成测试并汇总测试报告。"
            )
        }
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = now
        pipeline.logs.append(f"[{_timestamp()}] TestAgent 正在生成测试报告...")

        try:
            if self.engine is not None and StageType.TESTING in getattr(self.engine, "agents", {}):
                agent = self.engine.agents[StageType.TESTING]
            else:
                agent = TestAgent()

            # 将 worktree/project 路径写入 context，让 TestAgent 知道去哪儿跑 pytest
            ctx_dump = pipeline.context.model_dump()
            ctx_dump["project_path"] = _effective_project_path(pipeline)
            input_data = AgentInput(
                task_description="执行阶段: testing",
                context=ctx_dump,
                human_feedback=testing_stage.human_feedback,
            )
            output = await agent.execute(input_data)
            now = datetime.now()
            testing_stage.agent_output = output.result
            _apply_agent_metrics(testing_stage, output)
            testing_stage.status = (
                StageStatus.WAITING_HUMAN
                if output.needs_human_review or not output.success
                else StageStatus.COMPLETED
            )
            testing_stage.completed_at = now

            test_report = output.result.get("report") or output.details
            if isinstance(test_report, str) and test_report.strip():
                pipeline.context.test_report = test_report
                report_path = _write_project_doc(
                    _effective_docs_path(pipeline),
                    "test_report.md",
                    test_report,
                )
                pipeline.logs.append(f"[{_timestamp()}] 测试报告已写入: {report_path}")

            # 将测试文件写入 effective project path（worktree 或 project_path）。
            # 真实 TestAgent 在 _run_pytest 里也会自己写，这里保证 FakeAgent 场景下也落盘。
            test_files = output.result.get("test_files") or {}
            if isinstance(test_files, dict) and test_files:
                written_tests = _write_generated_code(
                    _effective_project_path(pipeline),
                    test_files,
                )
            else:
                written_tests = []

            pipeline.updated_at = now
            pipeline.status = (
                PipelineStatus.WAITING_HUMAN
                if output.needs_human_review or not output.success
                else PipelineStatus.PENDING
            )
            pipeline.logs.append(f"[{_timestamp()}] {output.summary}")
            _append_usage_log(pipeline, testing_stage)
            if written_tests:
                preview = "、".join(Path(path).name for path in written_tests[:5])
                pipeline.logs.append(f"[{_timestamp()}] 测试文件已写入: {preview}")
            if output.needs_human_review:
                pipeline.logs.append(f"[{_timestamp()}] 测试阶段进入人工确认，等待审批")
            _commit_stage(
                pipeline, StageType.TESTING,
                "test: add generated tests (stage4)",
                self.git,
            )
            await self.state_store.save(pipeline)
            if testing_stage.status == StageStatus.COMPLETED:
                await self._run_review(pipeline)
        except Exception as error:
            testing_stage.status = StageStatus.FAILED
            testing_stage.completed_at = datetime.now()
            pipeline.status = PipelineStatus.FAILED
            pipeline.error = f"[testing] {error}"
            pipeline.logs.append(f"[{_timestamp()}] 测试阶段失败: {error}")
            await self.state_store.save(pipeline)
            raise

    async def _run_review(self, pipeline: Pipeline) -> None:
        if len(pipeline.stages) < 5:
            return

        review_stage = pipeline.stages[4]
        now = datetime.now()
        review_stage.status = StageStatus.RUNNING
        review_stage.started_at = now
        review_stage.agent_output = {
            "text": (
                "## 正在生成代码评审结论\n\n"
                "测试阶段已完成，正在基于代码与测试报告生成评审意见。"
            )
        }
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = now
        pipeline.logs.append(f"[{_timestamp()}] ReviewAgent 正在生成评审报告...")

        try:
            if self.engine is not None and StageType.REVIEW in getattr(self.engine, "agents", {}):
                agent = self.engine.agents[StageType.REVIEW]
            else:
                agent = ReviewAgent()

            # 收集 diff 供 ReviewAgent 做基于变更的评审
            git_ctx = pipeline.context.git
            diff_text: str | None = None
            if git_ctx.enabled and git_ctx.worktree_path and git_ctx.base_commit:
                try:
                    diff_text = self.git.diff(
                        git_ctx.worktree_path,
                        base=git_ctx.base_commit,
                        head="HEAD",
                    )
                    diff_stats = self.git.diff_stats(
                        git_ctx.worktree_path,
                        base=git_ctx.base_commit,
                        head="HEAD",
                    )
                    git_ctx.diff_stats = diff_stats
                    pipeline.logs.append(
                        f"[{_timestamp()}] [Git] 本次变更: "
                        f"{diff_stats.get('files', 0)} 文件, "
                        f"+{diff_stats.get('insertions', 0)} / "
                        f"-{diff_stats.get('deletions', 0)}"
                    )
                except GitError:
                    pass

            # 将 diff 注入 context 供 ReviewAgent 使用
            ctx_dump = pipeline.context.model_dump()
            if diff_text:
                ctx_dump["code_diff"] = diff_text[:8000]  # 限制长度防止超 token
                pipeline.context.code_diff = diff_text[:8000]

            input_data = AgentInput(
                task_description="执行阶段: code_review",
                context=ctx_dump,
                human_feedback=review_stage.human_feedback,
            )
            output = await agent.execute(input_data)
            now = datetime.now()
            review_stage.agent_output = output.result
            _apply_agent_metrics(review_stage, output)
            review_stage.status = (
                StageStatus.WAITING_HUMAN
                if output.needs_human_review
                else StageStatus.COMPLETED
            )
            review_stage.completed_at = now

            review_report = output.result.get("report") or output.details
            if isinstance(review_report, str) and review_report.strip():
                pipeline.context.review_report = review_report
                review_path = _write_project_doc(
                    _effective_docs_path(pipeline),
                    "review_report.md",
                    review_report,
                )
                pipeline.logs.append(f"[{_timestamp()}] 评审报告已写入: {review_path}")

            pipeline.updated_at = now
            pipeline.status = (
                PipelineStatus.WAITING_HUMAN
                if output.needs_human_review
                else PipelineStatus.PENDING
            )
            pipeline.logs.append(f"[{_timestamp()}] {output.summary}")
            _append_usage_log(pipeline, review_stage)
            self._commit_stage(
                pipeline,
                StageType.REVIEW,
                self._default_commit_message(pipeline, StageType.REVIEW),
            )
            if output.needs_human_review:
                pipeline.logs.append(f"[{_timestamp()}] 代码评审进入人工确认，等待审批")
            await self.state_store.save(pipeline)
            if not output.needs_human_review:
                await self._run_delivery(pipeline)
        except Exception as error:
            review_stage.status = StageStatus.FAILED
            review_stage.completed_at = datetime.now()
            pipeline.status = PipelineStatus.FAILED
            pipeline.error = f"[code_review] {error}"
            pipeline.logs.append(f"[{_timestamp()}] 代码评审失败: {error}")
            await self.state_store.save(pipeline)
            raise

    async def _run_delivery(self, pipeline: Pipeline) -> None:
        if len(pipeline.stages) < 6:
            return

        delivery_stage = pipeline.stages[5]
        now = datetime.now()
        delivery_stage.status = StageStatus.RUNNING
        delivery_stage.started_at = now
        delivery_stage.agent_output = {
            "text": (
                "## 正在整理交付结果\n\n"
                "评审阶段已通过，正在生成交付清单与部署建议。"
            )
        }
        pipeline.status = PipelineStatus.RUNNING
        pipeline.updated_at = now
        pipeline.logs.append(f"[{_timestamp()}] DeliveryAgent 正在生成交付结果...")

        try:
            if self.engine is not None and StageType.DELIVERY in getattr(self.engine, "agents", {}):
                agent = self.engine.agents[StageType.DELIVERY]
            else:
                agent = DeliveryAgent()

            input_data = AgentInput(
                task_description="执行阶段: delivery",
                context=self._build_agent_input_context(pipeline),
                human_feedback=delivery_stage.human_feedback,
            )
            output = await agent.execute(input_data)
            now = datetime.now()
            delivery_stage.agent_output = output.result
            _apply_agent_metrics(delivery_stage, output)
            delivery_stage.status = (
                StageStatus.WAITING_HUMAN
                if output.needs_human_review
                else StageStatus.COMPLETED
            )
            delivery_stage.completed_at = now

            delivery_result = output.result.get("result") or output.details
            if isinstance(delivery_result, str) and delivery_result.strip():
                pipeline.context.delivery_result = delivery_result
                delivery_path = _write_project_doc(
                    _effective_docs_path(pipeline),
                    "delivery.md",
                    delivery_result,
                )
                pipeline.logs.append(f"[{_timestamp()}] 交付文档已写入: {delivery_path}")

            # 将 DeliveryAgent 产出的 PR 元数据同步到 GitContext
            git_ctx = pipeline.context.git
            if git_ctx.enabled:
                git_ctx.pr_title = output.result.get("pr_title") or git_ctx.pr_title
                git_ctx.pr_description = output.result.get("pr_description") or git_ctx.pr_description
                pr_cmd = output.result.get("pr_command")
                if pr_cmd:
                    git_ctx.pr_command = pr_cmd
                elif git_ctx.working_branch:
                    # 兜底：拼出一条可直接用的 gh pr create 命令
                    title = (git_ctx.pr_title or "feat: FlowState auto-generated").replace('"', '\\"')
                    git_ctx.pr_command = (
                        f'gh pr create --title "{title}" '
                        f'--head {git_ctx.working_branch} '
                        f'--body-file .flowstate/{pipeline.id}/docs/delivery.md'
                    )

            pipeline.updated_at = now
            pipeline.status = (
                PipelineStatus.WAITING_HUMAN
                if output.needs_human_review
                else PipelineStatus.COMPLETED
            )
            pipeline.logs.append(f"[{_timestamp()}] {output.summary}")
            _append_usage_log(pipeline, delivery_stage)
            self._commit_stage(
                pipeline,
                StageType.DELIVERY,
                self._default_commit_message(pipeline, StageType.DELIVERY),
            )
            if output.needs_human_review:
                pipeline.logs.append(f"[{_timestamp()}] 交付阶段进入人工确认，等待审批")
            else:
                pipeline.logs.append(f"[{_timestamp()}] 所有阶段已完成")
            await self.state_store.save(pipeline)
        except Exception as error:
            delivery_stage.status = StageStatus.FAILED
            delivery_stage.completed_at = datetime.now()
            pipeline.status = PipelineStatus.FAILED
            pipeline.error = f"[delivery] {error}"
            pipeline.logs.append(f"[{_timestamp()}] 交付阶段失败: {error}")
            await self.state_store.save(pipeline)
            raise
