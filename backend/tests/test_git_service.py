"""GitService 单元测试。

全部使用真实 git CLI 在 tmp_path 中跑，不 mock。
确保 CI 机器装有 git。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.git_service import (
    GitError,
    GitService,
    NestedRepoError,
    WorktreeBusyError,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def git_service() -> GitService:
    return GitService()


@pytest.fixture()
def empty_repo(tmp_path: Path, git_service: GitService) -> Path:
    """初始化一个仅有空提交的仓库。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    git_service.init_repo(repo, default_branch="main")
    git_service.write_default_gitignore(repo)
    git_service.baseline_commit(repo, message="chore: baseline")
    return repo


@pytest.fixture()
def repo_with_files(tmp_path: Path, git_service: GitService) -> Path:
    """初始化一个带初始文件的仓库。"""
    repo = tmp_path / "repo_with_files"
    repo.mkdir()
    (repo / "README.md").write_text("# hello\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    git_service.init_repo(repo, default_branch="main")
    git_service.write_default_gitignore(repo)
    git_service.baseline_commit(repo)
    return repo


# ---------------------------------------------------------------------------
# 探测
# ---------------------------------------------------------------------------


def test_is_available_true(git_service: GitService):
    assert git_service.is_available() is True


def test_is_git_repo(git_service: GitService, empty_repo: Path, tmp_path: Path):
    assert git_service.is_git_repo(empty_repo) is True
    assert git_service.is_git_repo(tmp_path) is False  # 父目录非 repo
    assert git_service.is_git_repo(tmp_path / "nope") is False  # 不存在


def test_find_repo_root(git_service: GitService, repo_with_files: Path):
    assert git_service.find_repo_root(repo_with_files) == repo_with_files
    assert git_service.find_repo_root(repo_with_files / "src") == repo_with_files


def test_find_enclosing_repo_detects_nested(
    git_service: GitService, repo_with_files: Path
):
    nested = repo_with_files / "subdir"
    nested.mkdir()
    enclosing = git_service.find_enclosing_repo(nested)
    assert enclosing == repo_with_files


def test_find_enclosing_repo_returns_none_when_self_is_repo(
    git_service: GitService, repo_with_files: Path
):
    assert git_service.find_enclosing_repo(repo_with_files) is None


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------


def test_init_repo_empty_dir(tmp_path: Path, git_service: GitService):
    target = tmp_path / "fresh"
    target.mkdir()
    git_service.init_repo(target, default_branch="main")
    assert (target / ".git").exists()
    assert git_service.current_branch(target) == "main"


def test_baseline_commit_with_files(tmp_path: Path, git_service: GitService):
    target = tmp_path / "withfiles"
    target.mkdir()
    (target / "a.txt").write_text("hello", encoding="utf-8")
    git_service.init_repo(target)
    git_service.write_default_gitignore(target)
    sha = git_service.baseline_commit(target)
    assert len(sha) == 40
    assert git_service.has_any_commit(target)
    files = git_service.changed_files(
        target, base=sha, head=sha
    )
    # head==base 时 diff 为空，但 commit 自身应可被 show
    info = git_service.show_commit(target, sha)
    assert info["sha"] == sha


def test_baseline_commit_empty_dir_uses_allow_empty(
    tmp_path: Path, git_service: GitService
):
    target = tmp_path / "empty"
    target.mkdir()
    git_service.init_repo(target)
    sha = git_service.baseline_commit(target)
    assert len(sha) == 40


def test_write_default_gitignore_preserves_existing(
    tmp_path: Path, git_service: GitService
):
    target = tmp_path / "g"
    target.mkdir()
    (target / ".gitignore").write_text("custom-line\n", encoding="utf-8")
    git_service.write_default_gitignore(target)
    text = (target / ".gitignore").read_text(encoding="utf-8")
    assert "custom-line" in text  # 没被覆盖


def test_ensure_gitignore_entry_appends(tmp_path: Path, git_service: GitService):
    target = tmp_path / "g2"
    target.mkdir()
    git_service.write_default_gitignore(target)
    git_service.ensure_gitignore_entry(target, "custom_dir/")
    text = (target / ".gitignore").read_text(encoding="utf-8")
    assert "custom_dir/" in text
    # 重复添加不会出现两次
    git_service.ensure_gitignore_entry(target, "custom_dir/")
    assert text.count("custom_dir/") <= (target / ".gitignore").read_text(
        encoding="utf-8"
    ).count("custom_dir/")


# ---------------------------------------------------------------------------
# Worktree
# ---------------------------------------------------------------------------


def test_add_and_remove_worktree(
    tmp_path: Path, git_service: GitService, repo_with_files: Path
):
    wt_path = repo_with_files / ".flowstate" / "wt1"
    base = git_service.head_commit(repo_with_files)
    git_service.add_worktree(
        repo_with_files, wt_path, branch="devflow/test-1", base=base
    )
    assert wt_path.exists()
    assert (wt_path / "README.md").exists()
    assert git_service.current_branch(wt_path) == "devflow/test-1"

    git_service.remove_worktree(repo_with_files, wt_path)
    assert not wt_path.exists()


def test_add_worktree_busy_when_path_exists(
    git_service: GitService, repo_with_files: Path
):
    wt_path = repo_with_files / ".flowstate" / "occupied"
    wt_path.mkdir(parents=True)
    base = git_service.head_commit(repo_with_files)
    with pytest.raises(WorktreeBusyError):
        git_service.add_worktree(
            repo_with_files, wt_path, branch="devflow/x", base=base
        )


def test_remove_worktree_is_idempotent(
    git_service: GitService, repo_with_files: Path
):
    # 移除不存在的 worktree 不应抛
    git_service.remove_worktree(
        repo_with_files, repo_with_files / "no-such-wt"
    )


def test_list_worktrees_includes_main(
    git_service: GitService, repo_with_files: Path
):
    paths = git_service.list_worktrees(repo_with_files)
    assert any(Path(p) == repo_with_files for p in paths)


# ---------------------------------------------------------------------------
# Commit / Diff / Reset
# ---------------------------------------------------------------------------


def test_commit_in_worktree_and_diff(
    git_service: GitService, repo_with_files: Path
):
    base = git_service.head_commit(repo_with_files)
    wt_path = repo_with_files / ".flowstate" / "wt-commit"
    git_service.add_worktree(repo_with_files, wt_path, branch="devflow/c1", base=base)

    # 在 worktree 中做改动
    (wt_path / "feature.py").write_text("def f(): pass\n", encoding="utf-8")
    assert git_service.has_changes(wt_path) is True

    sha = git_service.stage_and_commit(wt_path, "feat: add feature")
    assert sha is not None
    assert len(sha) == 40
    assert git_service.has_changes(wt_path) is False

    # diff 相对 base 应该有一个文件
    files = git_service.changed_files(wt_path, base=base, head=sha)
    assert "feature.py" in files
    stats = git_service.diff_stats(wt_path, base=base, head=sha)
    assert stats["files"] == 1
    assert stats["insertions"] >= 1
    diff = git_service.diff(wt_path, base=base, head=sha)
    assert "+def f(): pass" in diff

    # 主仓库工作区应该不变
    assert not (repo_with_files / "feature.py").exists()


def test_commit_without_changes_raises(
    git_service: GitService, repo_with_files: Path
):
    # 主仓库 HEAD 之上无变更
    with pytest.raises(GitError):
        git_service.commit(repo_with_files, "noop")


def test_stage_and_commit_returns_none_when_clean(
    git_service: GitService, repo_with_files: Path
):
    sha = git_service.stage_and_commit(repo_with_files, "noop")
    assert sha is None


def test_reset_hard_rewinds_worktree(
    git_service: GitService, repo_with_files: Path
):
    base = git_service.head_commit(repo_with_files)
    wt_path = repo_with_files / ".flowstate" / "wt-reset"
    git_service.add_worktree(repo_with_files, wt_path, branch="devflow/reset", base=base)

    (wt_path / "a.txt").write_text("A", encoding="utf-8")
    sha_a = git_service.stage_and_commit(wt_path, "feat: a")
    (wt_path / "b.txt").write_text("B", encoding="utf-8")
    sha_b = git_service.stage_and_commit(wt_path, "feat: b")

    assert sha_a and sha_b and sha_a != sha_b
    assert (wt_path / "b.txt").exists()

    git_service.reset_hard(wt_path, sha_a)
    assert git_service.head_commit(wt_path) == sha_a
    assert not (wt_path / "b.txt").exists()
    assert (wt_path / "a.txt").exists()
