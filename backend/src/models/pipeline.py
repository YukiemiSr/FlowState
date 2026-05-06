"""FlowState pipeline 数据模型"""

from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_HUMAN = "waiting_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageType(str, Enum):
    REQUIREMENT = "requirement_analysis"
    SOLUTION = "solution_design"
    CODING = "coding"
    TESTING = "testing"
    REVIEW = "code_review"
    DELIVERY = "delivery"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


class ApproveAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"


class GitMode(str, Enum):
    """Git 集成模式。

    DISABLED: 没有提供 project_path 或 git CLI 不可用，走"无 Git 模式"。
    WORKTREE: V1 默认模式——基于 git worktree 在隔离工作树中跑流水线。
    """

    DISABLED = "disabled"
    WORKTREE = "worktree"


class StageCommit(BaseModel):
    """每个 stage 完成时形成的 commit 锚点。

    用途：
    - 作为 reject / retry 时 reset_hard 的目标
    - 作为前端 Git 时间线展示数据
    """

    stage_type: StageType
    commit_sha: str
    commit_message: str
    committed_at: datetime
    files_changed: List[str] = []


class GitContext(BaseModel):
    """流水线的 Git 状态快照。"""

    mode: GitMode = GitMode.DISABLED
    enabled: bool = False

    # —— 仓库信息 ——
    repo_root: Optional[str] = None
    base_branch: Optional[str] = None
    base_commit: Optional[str] = None

    # —— Worktree 信息 ——
    worktree_path: Optional[str] = None
    working_branch: Optional[str] = None
    initialized: bool = False  # 仓库是否由 FlowState 自动 init

    # —— Stage commit 锚点 ——
    stage_commits: List[StageCommit] = []

    # —— 累计统计 ——
    total_files_changed: List[str] = []
    head_commit: Optional[str] = None
    diff_stats: Optional[Dict[str, Any]] = None

    # —— 交付元数据 ——
    pr_title: Optional[str] = None
    pr_description: Optional[str] = None
    pr_command: Optional[str] = None


class PipelineContext(BaseModel):
    """贯穿整个 Pipeline 的上下文"""
    project_path: str = ""
    project_summary: Optional[str] = None
    requirement_raw: str = ""
    requirement_doc: Optional[str] = None
    requirement_doc_path: Optional[str] = None
    solution_doc: Optional[str] = None
    solution_doc_path: Optional[str] = None
    solution_structured: Optional[Dict[str, Any]] = None
    generated_code: Optional[Dict[str, str]] = None
    test_report: Optional[str] = None
    review_report: Optional[str] = None
    delivery_result: Optional[str] = None
    code_diff: Optional[str] = None
    git: GitContext = Field(default_factory=GitContext)


class StageNode(BaseModel):
    """流水线中一个阶段节点"""
    model_config = ConfigDict(protected_namespaces=())

    stage_type: StageType
    status: StageStatus = StageStatus.PENDING
    agent_output: Optional[Dict[str, Any]] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model_name: Optional[str] = None
    human_feedback: Optional[str] = None
    human_approval: Optional[ApproveAction] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


class Pipeline(BaseModel):
    """完整研发流水线"""
    id: str = Field(
        default_factory=lambda: f"pipe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    title: str = ""
    status: PipelineStatus = PipelineStatus.PENDING
    stages: List[StageNode] = Field(default_factory=list)
    context: PipelineContext = Field(default_factory=PipelineContext)
    logs: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None
