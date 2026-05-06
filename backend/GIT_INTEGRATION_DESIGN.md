# FlowState · Git 集成设计方案（V1 本地）

> 版本：V1 · 仅本地 Git
> 作者：FlowState 团队
> 状态：待实施

---

## 1. 目标与非目标

### 1.1 目标

把当前 DeliveryAgent 仅产出"交付文案"的状态，升级为**真实的本地 Git 生命周期**：

- 每条 pipeline 在用户项目里建立**隔离的 git worktree**，所有 AI 产物都落在隔离环境中
- 各阶段完成时**真实 commit**，pipeline 跑完得到一条**可读、可追溯、可合并的分支**
- 提供**可观测的 Git 状态**给前端：当前分支、基线分支、各阶段 commit、变更文件、累计 diff
- 提供**reject / retry 的 Git 语义**：回退到上一阶段的 commit 锚点重做
- 输出**PR-ready 元数据**（标题/描述/分支名/`gh pr create` 命令），用户一键发起 PR

### 1.2 非目标（V1 不做）

- 不接 GitHub / GitLab / Gitea API
- 不自动 `git push`
- 不自动创建 MR/PR
- 不做远端分支同步
- 不做 fork / multi-remote 处理
- 不做合并冲突自动解决

> 上述能力作为 V2 增量能力，不在本设计内。

---

## 2. 核心决策（讨论结论摘要）

| 决策项 | 结论 | 理由 |
|---|---|---|
| 隔离方式 | **git worktree** | 不污染用户主工作树，无"工作区必须干净"前置条件，reject 回退干净 |
| Worktree 位置 | `<project_path>/.flowstate/worktrees/<pipeline-id>/` | 用户已在前端指定 path，仓库内便于观察 |
| Commit 粒度 | **分阶段提交**（每个 stage 一次 commit） | `git log` 本身讲故事，演示强；reject 回退靠 commit 锚点 |
| 非 Git 仓库 | **自动 `git init` + baseline 提交** | 避免"差一步 init"打断流程；含父仓库探测保护 |
| Doc 落点 | `<worktree>/.flowstate/<pipeline-id>/docs/` | 与代码产物隔离；废弃当前 `<repo>/main/` 命名 |
| 远端动作 | **全部禁用** | V1 范围内全关；产出 `gh pr create` 命令字符串供用户手动执行 |

---

## 3. 系统架构

### 3.1 模块划分

```
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI 路由层                           │
│       routers/pipelines.py    routers/git.py（新）            │
└────────────┬─────────────────────────────┬───────────────────┘
             │                             │
             ▼                             ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│    PipelineService       │   │     GitService（新）      │
│  （编排时机 + 阶段触发）   │──▶│  （所有 git 命令封装）    │
└──────────────────────────┘   └──────────────────────────┘
             │                             │
             ▼                             ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│  Agent (Stage 1..6)      │   │  subprocess: git CLI     │
└──────────────────────────┘   └──────────────────────────┘
             │
             ▼
┌──────────────────────────┐
│  StateStore (持久化)      │
└──────────────────────────┘
```

### 3.2 职责边界

- **GitService**：纯封装 git 命令，无业务语义。输入路径与参数，输出结构化结果或抛 `GitError`。**不知道** stage 是什么。
- **PipelineService**：决定**在哪个时机**调用 GitService。stage 间 hook 都在它里面。
- **Agent**：产出文档/代码/diff 评审等**内容**，不直接执行 git。
- **新 router `routers/git.py`**：暴露只读 git 状态（前端展示）和"生成 PR 命令"等动作。

---

## 4. 数据模型变更

### 4.1 新增 `GitContext`

```python
# src/models/pipeline.py

class GitMode(str, Enum):
    DISABLED = "disabled"        # project_path 不可用，或显式关闭
    WORKTREE = "worktree"        # V1 默认模式

class StageCommit(BaseModel):
    """每个 stage 完成时的 commit 锚点，用于 reject 回退。"""
    stage_type: StageType
    commit_sha: str
    commit_message: str
    committed_at: datetime
    files_changed: list[str] = []

class GitContext(BaseModel):
    mode: GitMode = GitMode.DISABLED
    enabled: bool = False

    # 仓库信息
    repo_root: Optional[str] = None         # 用户原仓库根
    base_branch: Optional[str] = None       # 派生基线（默认 main / master / 当前 HEAD）
    base_commit: Optional[str] = None       # 基线起点 sha

    # Worktree 信息
    worktree_path: Optional[str] = None     # <repo>/.flowstate/worktrees/<id>/
    working_branch: Optional[str] = None    # devflow/<id>-<slug>
    initialized: bool = False               # 仓库是否由 FlowState 自动 init

    # Stage commit 锚点（reject 回退用）
    stage_commits: list[StageCommit] = []

    # 累计统计
    total_files_changed: list[str] = []
    head_commit: Optional[str] = None       # worktree 当前 HEAD
    diff_stats: Optional[dict] = None       # {"insertions": X, "deletions": Y, "files": Z}

    # 交付元数据
    pr_title: Optional[str] = None
    pr_description: Optional[str] = None
    pr_command: Optional[str] = None        # 现成的 gh pr create 命令
```

### 4.2 `PipelineContext` 增加字段

```python
class PipelineContext(BaseModel):
    # ... 现有字段 ...
    git: GitContext = GitContext()
```

### 4.3 `PipelineContext.project_path` 语义微调

| 阶段 | `project_path` 指向 |
|---|---|
| Pipeline 创建前 | 用户原项目根 |
| Git 初始化后 | **依然指向用户原根**（不变） |
| Stage 执行时 | Agent 收到的 `project_path` = `git.worktree_path`（如果启用 git）|

> Agent 不需要知道 worktree 概念，只需要拿到一个"可写的项目目录"。PipelineService 在调用 agent 时**替换** `project_path`，对 agent 透明。

---

## 5. GitService API 设计

```python
# src/services/git_service.py（新文件）

class GitError(Exception): ...
class NotARepoError(GitError): ...
class NestedRepoError(GitError): ...
class WorktreeBusyError(GitError): ...

class GitService:
    """纯命令封装，无状态，可并发。"""

    # —— 仓库探测 ——
    def is_git_repo(self, path: Path) -> bool: ...
    def find_repo_root(self, path: Path) -> Path | None: ...
    def find_enclosing_repo(self, path: Path) -> Path | None:
        """从 path 向上找最近的 .git；用于检测嵌套仓库风险。"""

    # —— 仓库初始化 ——
    def init_repo(self, path: Path, *, default_branch: str = "main") -> None: ...
    def write_default_gitignore(self, path: Path) -> None: ...
    def baseline_commit(self, path: Path, message: str = "chore: flowstate baseline") -> str:
        """git add . && git commit；返回 sha。空仓库时也允许空提交。"""

    # —— 分支与 worktree ——
    def current_branch(self, repo: Path) -> str: ...
    def head_commit(self, repo: Path) -> str: ...
    def add_worktree(
        self, repo: Path, worktree_path: Path, branch: str, *, base: str
    ) -> None:
        """git worktree add <worktree_path> -b <branch> <base>"""
    def remove_worktree(self, repo: Path, worktree_path: Path, *, force: bool = False) -> None: ...
    def delete_branch(self, repo: Path, branch: str, *, force: bool = True) -> None: ...

    # —— 提交 ——
    def stage_all(self, worktree: Path) -> None: ...
    def commit(
        self, worktree: Path, message: str, *, allow_empty: bool = False
    ) -> str:
        """返回新 commit 的 sha；无变更且 allow_empty=False 时抛错。"""
    def has_changes(self, worktree: Path) -> bool: ...

    # —— 查询 ——
    def diff(self, worktree: Path, *, base: str, head: str = "HEAD") -> str:
        """返回 unified diff 文本。"""
    def diff_stats(self, worktree: Path, *, base: str, head: str = "HEAD") -> dict:
        """{'files': N, 'insertions': X, 'deletions': Y}"""
    def changed_files(self, worktree: Path, *, base: str, head: str = "HEAD") -> list[str]: ...
    def show_commit(self, worktree: Path, sha: str) -> dict: ...

    # —— 回退 ——
    def reset_hard(self, worktree: Path, ref: str) -> None: ...
```

### 5.1 实现要点

- 全部走 `subprocess.run(["git", ...], cwd=..., capture_output=True, check=True)`，包装 stderr 到 `GitError`
- 不引入第三方库（如 GitPython）；保持依赖最小
- 统一字符编码 utf-8
- 命令超时（默认 30s）

---

## 6. 流水线钩子点（关键）

### 6.1 Pipeline 创建时（`PipelineService.create_pipeline`）

```
1. 解析 project_path（已有逻辑）
2. 调用 GitService 探测：
   case A: project_path 已是 git repo
       → git.repo_root = repo_root
       → git.base_branch = current_branch（HEAD）
       → git.base_commit = HEAD sha
       → mode = WORKTREE, enabled = True
   case B: project_path 不是 repo，但父目录是
       → 抛 NestedRepoError，前端 400 提示用户改路径
   case C: project_path 不是 repo，目录为空或非空
       → git_service.write_default_gitignore(path)
       → git_service.init_repo(path, default_branch="main")
       → git_service.baseline_commit(path)  ← 把现有文件作为基线
       → git.initialized = True
       → 后续同 case A
   case D: project_path 为空字符串
       → mode = DISABLED, enabled = False
       → 走"无 Git 模式"，与现状等价

3. 若 enabled:
   - 计算 working_branch 名: f"devflow/{pipeline.id}-{slugify(title)[:24]}"
   - worktree_path = <repo_root>/.flowstate/worktrees/<pipeline-id>/
   - git_service.add_worktree(repo_root, worktree_path, working_branch, base=base_commit)
   - 把 .flowstate/ 加入用户原仓库的 .gitignore（若未加）
   - pipeline.context.git 写入完整信息
   - pipeline.logs 追加 "已创建工作分支 X，工作目录 Y"
```

### 6.2 每个 Stage 执行前

PipelineService 在调用 agent 之前注入：

```python
effective_project_path = (
    pipeline.context.git.worktree_path
    if pipeline.context.git.enabled
    else pipeline.context.project_path
)
input_data = AgentInput(
    task_description=...,
    context={**pipeline.context.model_dump(), "project_path": effective_project_path},
    ...
)
```

> Agent 代码**不需要改**。它依旧把 `requirements.md` 写到 `project_path/...`，只是这个路径已经被替换成 worktree。

### 6.3 每个 Stage 完成后（成功 commit）

```python
async def _commit_stage(pipeline, stage_type, message_for_stage):
    git = pipeline.context.git
    if not git.enabled:
        return
    if not git_service.has_changes(git.worktree_path):
        return  # stage 没有产生任何文件变更，跳过 commit
    git_service.stage_all(git.worktree_path)
    sha = git_service.commit(git.worktree_path, message_for_stage)
    files = git_service.changed_files(git.worktree_path, base=git.base_commit, head=sha)
    git.stage_commits.append(StageCommit(
        stage_type=stage_type,
        commit_sha=sha,
        commit_message=message_for_stage,
        committed_at=datetime.now(),
        files_changed=files,
    ))
    git.head_commit = sha
    git.diff_stats = git_service.diff_stats(git.worktree_path, base=git.base_commit)
    git.total_files_changed = git_service.changed_files(
        git.worktree_path, base=git.base_commit
    )
    pipeline.logs.append(f"[stage_type] commit {sha[:7]}: {message_for_stage}")
```

### 6.4 各 stage 的 commit message 模板

| Stage | Commit message |
|---|---|
| requirement_analysis | `docs(flowstate): structured requirements` |
| solution_design | `docs(flowstate): solution design` |
| coding | `feat(flowstate): implement <短标题>` |
| testing | `test(flowstate): add tests and report` |
| code_review | （**不 commit**——评审是注释而非代码产物，写到 docs 但不计入 git；或单独可选） |
| delivery | `chore(flowstate): finalize delivery` + 写 `pr.md` |

> 决议：**review 阶段不 commit**。如果用户希望评审报告进入历史，可以在 settings 里开"commit review report"开关（默认关）。

### 6.5 Code Review 阶段：基于 diff

Stage 5 进入时，PipelineService 调用：

```python
diff_text = git_service.diff(
    git.worktree_path,
    base=git.base_commit,
    head=git.head_commit,
)
```

将 `diff_text` 注入 ReviewAgent 的 input context（新增字段 `code_diff`）。`ReviewAgent` 的 prompt 改为"基于以下变更 diff 做评审"，而非"基于全文"。

### 6.6 Delivery 阶段

DeliveryAgent 仍负责生成 PR 元信息（已经在做了）。完成后：

```python
git.pr_title = delivery.pr_title
git.pr_description = delivery.pr_description
git.pr_command = (
    f'gh pr create --title "{shlex.quote(git.pr_title)}" '
    f'--body-file .flowstate/{pipeline.id}/docs/pr.md '
    f'--head {git.working_branch} --base {git.base_branch}'
)
# 写 pr.md 到 worktree 的 docs 目录
write_doc(worktree, "pr.md", build_pr_body(delivery))
# 最终 commit
_commit_stage(pipeline, DELIVERY, "chore(flowstate): finalize delivery")
```

### 6.7 Reject / Retry

**Reject（在某个 waiting_human stage 上拒绝）：**

```python
async def reject_stage(...):
    # ... 现有逻辑 ...
    if pipeline.context.git.enabled:
        # 找到该 stage 的"上一阶段"的 commit 锚点
        prev = _find_prev_commit_anchor(pipeline, stage_index)
        target_sha = prev.commit_sha if prev else pipeline.context.git.base_commit
        git_service.reset_hard(pipeline.context.git.worktree_path, target_sha)
        # 截断 stage_commits（移除当前 stage 及其后的锚点）
        pipeline.context.git.stage_commits = [
            sc for sc in pipeline.context.git.stage_commits
            if _stage_order(sc.stage_type) < stage_index
        ]
        pipeline.context.git.head_commit = target_sha
```

**Retry（从失败 stage 重跑）：**

同 reject：reset 到上一锚点，从那里开始重跑。

---

## 7. Doc 文件路径迁移

### 7.1 旧路径

```
<project_path>/main/
├── requirements.md
├── solution.md
├── test_report.md
├── review_report.md
└── delivery.md
```

### 7.2 新路径

```
<worktree>/.flowstate/<pipeline-id>/docs/
├── requirements.md
├── solution.md
├── test_report.md
├── review_report.md
├── delivery.md
└── pr.md           ← 新增（DeliveryAgent 生成 PR 描述全文）
```

### 7.3 迁移点

`service.py::_write_project_doc(project_path, filename, content)` 改为：

```python
def _write_project_doc(pipeline, filename, content) -> Path:
    """写入流水线 doc。"""
    if pipeline.context.git.enabled:
        base = Path(pipeline.context.git.worktree_path) / ".flowstate" / pipeline.id / "docs"
    else:
        # 无 git 模式：放在用户 project_path/.flowstate/<id>/docs/
        base = Path(pipeline.context.project_path) / ".flowstate" / pipeline.id / "docs"
    base.mkdir(parents=True, exist_ok=True)
    target = base / filename
    target.write_text(content, encoding="utf-8")
    return target
```

调用方签名从 `(project_path, filename, content)` 改为 `(pipeline, filename, content)`。

> 这次一并废弃 `<repo>/main/` 命名。

---

## 8. Settings 扩展

`flowstate.config.json` 顶层新增 `git` 段：

```json
{
  "git": {
    "enabled": true,
    "auto_init_if_missing": true,
    "default_base_branch": "main",
    "branch_naming_template": "devflow/{pipeline_id}-{slug}",
    "commit_per_stage": true,
    "commit_review_report": false,
    "auto_push": false,
    "auto_create_pr": false
  }
}
```

V1 中 `auto_push` / `auto_create_pr` 保留字段但**永远为 false**（前端可见、不可点）。V2 接入。

---

## 9. API 变更

### 9.1 现有路由：行为补充，签名不变

- `POST /api/v1/pipelines`：内部多一步 git 准备；失败时返回 400 + `detail` 携带 git 错误信息（如 `NESTED_REPO`）
- `POST /api/v1/pipelines/{id}/approve` / `reject`：内部触发 stage commit / reset
- `GET /api/v1/pipelines/{id}`：响应体的 `pipeline.context.git` 现在有真实数据

### 9.2 新增路由 `routers/git.py`

```
GET  /api/v1/pipelines/{id}/git/status      # 当前 GitContext 快照
GET  /api/v1/pipelines/{id}/git/diff        # 累计 diff 文本（base..HEAD）
GET  /api/v1/pipelines/{id}/git/diff/{stage} # 单 stage 的 diff
GET  /api/v1/pipelines/{id}/git/log          # stage commits 列表
GET  /api/v1/pipelines/{id}/git/pr-command  # 现成的 gh pr create 命令
DELETE /api/v1/pipelines/{id}/git           # 清理 worktree + 删分支（用于放弃）
```

只读为主，写操作仅 DELETE（用于"我不要了"）。

---

## 10. 前端展示要点（不在本设计实现，但留接口）

详情页新增 **Git 面板**：

- 仓库路径 / 基线分支 / 工作分支
- Worktree 路径（带"在 Finder 中打开"按钮）
- Stage commits 时间线（每行：stage 名 / sha / message / 文件数）
- 累计 diff 统计（X 文件 / +Y / -Z）
- "查看完整 diff"按钮 → 调 `/git/diff`
- "生成 PR 命令"区块：显示 `git.pr_command`，一键复制

CheckpointReview 页（review 阶段）需要把 diff 渲染出来（建议用现有 monaco 或简单的 `<pre>` + 高亮）。

---

## 11. 错误处理与边界

| 情形 | 行为 |
|---|---|
| `git` 命令未安装 | 启动时检测，若启用 git 集成则报错 fail-fast |
| project_path 父目录是 repo | 创建 pipeline 时返回 400 `NESTED_REPO` |
| Worktree 路径已存在（残留） | 创建时尝试清理；清理失败则报错让用户手动删 |
| Stage 没产生任何文件变更 | 跳过 commit（不抛错） |
| Stage commit 失败 | stage 标记 FAILED，pipeline FAILED，logs 记录 git stderr |
| Reject 时 reset 失败 | logs 警告但**不阻塞** reject 本身（业务状态优先） |
| 删除 pipeline 时 worktree 还在 | 异步清理 worktree + 删分支；失败仅日志 |
| 用户手动改了 worktree 内文件 | 我们不防御，下个 stage commit 时一并入提交 |

---

## 12. 测试计划

### 12.1 单元测试（`tests/test_git_service.py`）

- `init_repo` + `baseline_commit` 在空目录、有文件目录两种情况
- 嵌套仓库探测
- `add_worktree` / `remove_worktree` / 分支删除
- `commit` 在有/无变更时的行为
- `diff` / `diff_stats` 正确返回
- `reset_hard` 行为

> 在 `tmp_path` 下用真实 git 命令跑，不 mock。

### 12.2 集成测试（`tests/test_pipeline_git.py`）

- 端到端：创建 pipeline（空目录）→ 跑完 6 阶段 → 校验 worktree 内有 N 个 stage commit、分支名正确、`pr.md` 存在
- 已有 git repo 场景：基线分支被正确识别
- 嵌套仓库场景：返回 400
- Reject 场景：reset 到上一锚点，stage_commits 被截断
- Disabled 模式（project_path 空）：行为退化到现状

### 12.3 手动验证清单

- [ ] 在 FlowState 自身仓库上跑一次 pipeline，确认主工作树纹丝不动
- [ ] worktree 内 `git log --oneline` 显示 5 条规整的 stage commit
- [ ] `gh pr create` 命令可直接复制执行
- [ ] reject + 重做后 git log 干净，无垃圾 commit

---

## 13. 实施顺序（建议拆 PR）

| # | 改动 | 可独立交付 |
|---|---|---|
| 1 | 新建 `GitService`，纯命令封装 + 单测 | ✅ |
| 2 | `models/pipeline.py` 加 `GitContext` 和 `StageCommit` | ✅ |
| 3 | `PipelineService.create_pipeline` 接入 git 准备（仓库探测/init/worktree 创建） | ✅ |
| 4 | 各 stage 执行前替换 `project_path` 为 worktree 路径；执行后 commit | ✅ |
| 5 | Doc 路径迁移到 `.flowstate/<id>/docs/`，废弃 `main/` | ✅（可与 4 合并） |
| 6 | Reject / retry 接入 reset_hard | ✅ |
| 7 | ReviewAgent 接入 diff 输入 | ✅ |
| 8 | `routers/git.py` 只读接口 + DELETE 清理 | ✅ |
| 9 | DeliveryAgent 产出 `pr.md` + `pr_command` | ✅ |
| 10 | 前端 Git 面板 | 独立 |

每一步都能落到 main 不破坏现有功能（`mode = DISABLED` 是安全 fallback）。

---

## 14. 风险与开放问题

- **风险 1**：worktree 残留。如果进程崩溃，`.flowstate/worktrees/<id>` 可能没清理。**对策**：启动时扫描 `.flowstate/worktrees/`，对照 state_store 中已删除的 pipeline 做孤儿清理。
- **风险 2**：用户原仓库基线分支不是 main/master（如 `develop`）。**对策**：默认取 `git symbolic-ref refs/remotes/origin/HEAD` 或当前 HEAD 名，而非硬编码 `main`。
- **风险 3**：`.flowstate/` 加入 `.gitignore` 时若用户已 stage 过该路径会被忽略不掉。**对策**：写 `.gitignore` 后检测一次 `git check-ignore`，必要时提醒用户。
- **开放问题 1**：是否需要"pipeline 完成后自动归档 worktree"（移到 `.flowstate/archive/`）？V1 暂不做，pipeline 删除时一起清理。
- **开放问题 2**：分支名 slug 含中文怎么办？建议 slugify 时直接丢弃非 ASCII，仅保留 pipeline_id 即可（`devflow/pipe_20260506_103022`）。

---

## 15. 一句话总结

V1 把"AI 生成的代码到底落在哪里"这件事**从用户工作树搬到隔离的 git worktree**，每个 stage 真实 commit，跑完得到一个干净的 `devflow/<id>` 分支和一行就能发 PR 的 `gh` 命令——既安全，又把 git log 本身变成 demo 素材。
