"""Git 状态只读接口 + Worktree 清理接口。

路由前缀：/api/v1/pipelines/{pipeline_id}/git
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.service import PipelineService
from src.models.pipeline import GitContext

router = APIRouter()


def _get_service() -> PipelineService:
    from src.api.app import get_pipeline_service  # 避免循环导入
    return get_pipeline_service()


# ---------------------------------------------------------------------------
# GET /api/v1/pipelines/{pipeline_id}/git
# ---------------------------------------------------------------------------


@router.get("/{pipeline_id}/git", summary="获取流水线 Git 状态")
async def get_git_status(
    pipeline_id: str,
    service: PipelineService = Depends(_get_service),
) -> dict:
    """返回该流水线的 GitContext 快照，包括分支信息、stage commit 列表、diff stats 等。"""
    pipeline = await service.get_pipeline(pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}")

    git_ctx: GitContext = pipeline.context.git
    return {
        "pipeline_id": pipeline_id,
        "git": git_ctx.model_dump(),
    }


# ---------------------------------------------------------------------------
# DELETE /api/v1/pipelines/{pipeline_id}/git/worktree
# ---------------------------------------------------------------------------


@router.delete("/{pipeline_id}/git/worktree", summary="清理流水线 Worktree")
async def cleanup_worktree(
    pipeline_id: str,
    service: PipelineService = Depends(_get_service),
) -> dict:
    """移除该流水线关联的 git worktree 及对应功能分支。

    仅在 pipeline 已 COMPLETED / CANCELLED / FAILED 时允许操作。
    """
    from src.models.pipeline import PipelineStatus

    pipeline = await service.get_pipeline(pipeline_id)
    if pipeline is None:
        raise HTTPException(status_code=404, detail=f"Pipeline not found: {pipeline_id}")

    terminal_statuses = {PipelineStatus.COMPLETED, PipelineStatus.CANCELLED, PipelineStatus.FAILED}
    if pipeline.status not in terminal_statuses:
        raise HTTPException(
            status_code=409,
            detail="只有已完成、已取消或已失败的流水线才可清理 Worktree",
        )

    git_ctx = pipeline.context.git
    if not git_ctx.enabled or not git_ctx.worktree_path:
        return {"removed": False, "reason": "该流水线未启用 Git 集成"}

    from src.services.git_service import GitError, get_git_service
    from pathlib import Path

    git = get_git_service()
    repo_root = git_ctx.repo_root
    wt_path = git_ctx.worktree_path
    branch = git_ctx.working_branch
    errors: list[str] = []

    # 移除 worktree
    try:
        if repo_root:
            git.remove_worktree(repo_root, wt_path)
    except GitError as exc:
        errors.append(f"remove_worktree: {exc}")

    # 删除功能分支（仅在 pipeline CANCELLED / FAILED 时删除，COMPLETED 保留供 PR）
    branch_deleted = False
    if pipeline.status != PipelineStatus.COMPLETED and branch and repo_root:
        try:
            git.delete_branch(repo_root, branch, force=True)
            branch_deleted = True
        except GitError as exc:
            errors.append(f"delete_branch: {exc}")

    # 更新 GitContext
    git_ctx.worktree_path = None
    await service.state_store.save(pipeline)

    return {
        "removed": True,
        "worktree_path": wt_path,
        "branch": branch,
        "branch_deleted": branch_deleted,
        "errors": errors,
    }
