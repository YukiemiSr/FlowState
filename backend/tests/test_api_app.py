import asyncio
import base64
import os
from pathlib import Path

from fastapi.testclient import TestClient

from src.agents.base_agent import AgentInput, AgentOutput
from src.llm_client import LLMResponse
from src.agents.requirement_agent import RequirementAgent
from src.api.app import create_app
from src.bootstrap import create_engine
from src.models.pipeline import PipelineStatus, StageStatus, StageType
from src.services.git_service import GitError
from src.store.state_store import StateStore


class FakeRequirementAgent:
    async def execute(self, input_data):
        project_path = input_data.context.get("project_path", "")
        project_summary = input_data.context.get("project_summary", "")
        requirement_raw = input_data.context.get("requirement_raw", "")
        document = f"""# 需求文档

## 原始需求
{requirement_raw}

## 项目目录
{project_path}

## 项目摘要
{project_summary}

```json
{{
  "modules": [
    {{"name": "收藏功能", "priority": "P0"}},
    {{"name": "列表展示", "priority": "P1"}}
  ]
}}
```"""
        return AgentOutput(
            success=True,
            result={
                "document": document,
                "raw_requirement": requirement_raw,
                "modules": [
                    {"name": "收藏功能", "priority": "P0"},
                    {"name": "列表展示", "priority": "P1"},
                ],
            },
            summary="需求分析完成，识别出 2 个功能模块",
            details=document,
            needs_human_review=True,
            token_usage={"prompt_tokens": 120, "completion_tokens": 280, "total_tokens": 400},
            model="fake-requirement-model",
        )


class FakeSolutionAgent:
    async def execute(self, input_data):
        requirement_doc = input_data.context.get("requirement_doc", "")
        document = f"""# 技术方案文档

## 整体架构
采用前后端分离架构，后端使用 FastAPI，前端使用 React。

## 核心 API 设计
### 创建计划
- 方法：POST
- 路径：`/plans`

## 需求背景
{requirement_doc}

```json
{{
  "resolved_stack": {{
    "frontend": "React Native",
    "backend": "FastAPI",
    "database": "SQLite",
    "state_management": "Zustand",
    "notifications": "Expo Notifications",
    "testing": "pytest + vitest"
  }},
  "file_plan": [
    {{
      "path": "app/main.py",
      "layer": "backend",
      "purpose": "FastAPI 入口",
      "must_generate": true
    }},
    {{
      "path": "requirements.txt",
      "layer": "config",
      "purpose": "Python 依赖定义",
      "must_generate": true
    }}
  ],
  "api_design": [
    {{
      "name": "创建计划",
      "method": "POST",
      "path": "/plans",
      "description": "创建每日计划",
      "request": "标题、日期、优先级",
      "response": "计划详情"
    }}
  ]
}}
```"""
        return AgentOutput(
            success=True,
            result={
                "design": document,
                "architecture": "前后端分离架构",
                "tech_stack": ["FastAPI", "React", "SQLite"],
                "resolved_stack": {
                    "frontend": "React Native",
                    "backend": "FastAPI",
                    "database": "SQLite",
                    "state_management": "Zustand",
                    "notifications": "Expo Notifications",
                    "testing": "pytest + vitest",
                },
                "file_plan": [
                    {
                        "path": "app/main.py",
                        "layer": "backend",
                        "purpose": "FastAPI 入口",
                        "must_generate": True,
                    },
                    {
                        "path": "requirements.txt",
                        "layer": "config",
                        "purpose": "Python 依赖定义",
                        "must_generate": True,
                    },
                ],
                "structured_solution": {
                    "resolved_stack": {
                        "frontend": "React Native",
                        "backend": "FastAPI",
                        "database": "SQLite",
                        "state_management": "Zustand",
                        "notifications": "Expo Notifications",
                        "testing": "pytest + vitest",
                    },
                    "file_plan": [
                        {
                            "path": "app/main.py",
                            "layer": "backend",
                            "purpose": "FastAPI 入口",
                            "must_generate": True,
                        },
                        {
                            "path": "requirements.txt",
                            "layer": "config",
                            "purpose": "Python 依赖定义",
                            "must_generate": True,
                        },
                    ],
                },
                "api_design": [
                    {
                        "name": "创建计划",
                        "method": "POST",
                        "path": "/plans",
                        "description": "创建每日计划",
                        "request": "标题、日期、优先级",
                        "response": "计划详情",
                    }
                ],
            },
            summary="技术方案已生成：1 个核心接口，3 项技术选型",
            details=document,
            needs_human_review=True,
            token_usage={"prompt_tokens": 160, "completion_tokens": 440, "total_tokens": 600},
            model="fake-solution-model",
        )


class FakeCodeAgent:
    async def execute(self, input_data):
        structured_solution = input_data.context.get("solution_structured", {})
        file_plan = structured_solution.get("file_plan", [])
        assert any(item.get("path") == "app/main.py" for item in file_plan)
        assert structured_solution.get("resolved_stack", {}).get("backend") == "FastAPI"
        return AgentOutput(
            success=True,
            result={
                "files": {
                    "app/main.py": "from fastapi import FastAPI\n\napp = FastAPI()\n",
                    "requirements.txt": "fastapi\nuvicorn\n",
                },
                "structure": ["app/main.py", "requirements.txt"],
                "language": "python",
                "framework": "fastapi",
            },
            summary="代码生成完成，共 2 个文件",
            details="  ✅ app/main.py\n  ✅ requirements.txt",
            needs_human_review=False,
            token_usage={"prompt_tokens": 220, "completion_tokens": 780, "total_tokens": 1000},
            model="fake-code-model",
        )


class FakeTestAgent:
    async def execute(self, input_data):
        generated_code = input_data.context.get("generated_code", {})
        assert "app/main.py" in generated_code
        return AgentOutput(
            success=True,
            result={
                "test_files": {
                    "tests/test_app.py": (
                        "from fastapi.testclient import TestClient\n"
                        "from app.main import app\n\n"
                        "def test_health():\n"
                        "    client = TestClient(app)\n"
                        "    response = client.get('/')\n"
                        "    assert response.status_code in {200, 404}\n"
                    )
                },
                "test_results": {"total": 1, "passed": 1, "failed": 0, "coverage": 85},
                "report": "## 测试报告\n\n- 总计: 1 项测试\n- 通过: 1 ✅\n- 失败: 0 ❌\n- 代码覆盖率: 85%",
                "pass_rate": "1/1",
            },
            summary="测试完成：1/1 通过",
            details="## 测试报告\n\n- 总计: 1 项测试\n- 通过: 1 ✅\n- 失败: 0 ❌\n- 代码覆盖率: 85%",
            needs_human_review=False,
            token_usage={"prompt_tokens": 90, "completion_tokens": 110, "total_tokens": 200},
            model="fake-test-model",
        )


class FakeFailingTestAgent:
    async def execute(self, input_data):
        generated_code = input_data.context.get("generated_code", {})
        assert "app/main.py" in generated_code
        return AgentOutput(
            success=False,
            result={
                "test_files": {
                    "tests/test_app.py": (
                        "def test_health():\n"
                        "    assert False, 'health endpoint missing'\n"
                    )
                },
                "test_results": {
                    "total": 1,
                    "passed": 0,
                    "failed": 1,
                    "coverage": 40,
                    "errors": ["health endpoint missing"],
                },
                "report": "## 测试报告\n\n- 总计: 1 项测试\n- 通过: 0 ✅\n- 失败: 1 ❌\n- 代码覆盖率: 40%\n\n### 错误详情\n- health endpoint missing",
                "pass_rate": "0/1",
            },
            summary="测试完成：0/1 通过",
            details="## 测试报告\n\n- 总计: 1 项测试\n- 通过: 0 ✅\n- 失败: 1 ❌\n- 代码覆盖率: 40%",
            needs_human_review=True,
            token_usage={"prompt_tokens": 80, "completion_tokens": 100, "total_tokens": 180},
            model="fake-test-model",
        )


class FakeReviewAgent:
    async def execute(self, input_data):
        test_report = input_data.context.get("test_report", "")
        assert "测试报告" in test_report
        return AgentOutput(
            success=True,
            result={
                "review": {
                    "score": 92,
                    "summary": "整体实现清晰，建议补充边界测试。",
                    "strengths": ["结构清晰", "接口定义明确"],
                    "issues": [],
                    "suggestions": ["补充异常路径断言"],
                },
                "report": "## 代码评审报告\n\n**评分: 92/100**\n\n- 建议补充异常路径断言",
                "issues": [],
                "score": 92,
            },
            summary="评审完成：评分 92/100，发现 0 个问题",
            details="## 代码评审报告\n\n**评分: 92/100**\n\n- 建议补充异常路径断言",
            needs_human_review=True,
            token_usage={"prompt_tokens": 70, "completion_tokens": 130, "total_tokens": 200},
            model="fake-review-model",
        )


class FakeDeliveryAgent:
    async def execute(self, input_data):
        review_report = input_data.context.get("review_report", "")
        assert "代码评审报告" in review_report
        return AgentOutput(
            success=True,
            result={
                "pr_title": "feat: 完成自律应用核心链路",
                "pr_description": "包含需求、方案、代码、测试与评审结果。",
                "branch": "feature/self-discipline-app",
                "commit_message": "feat: deliver self-discipline app pipeline output",
                "deployment_command": "docker compose up -d",
                "changes": 3,
                "files_changed": ["app/main.py", "requirements.txt", "tests/test_app.py"],
                "pr_number": 88,
                "result": "## 📦 交付汇总\n\n**PR 编号:** #88\n**标题:** feat: 完成自律应用核心链路",
            },
            summary="交付就绪：生成 PR #88，变更 3 个文件",
            details="## 📦 交付汇总\n\n**PR 编号:** #88\n**标题:** feat: 完成自律应用核心链路",
            needs_human_review=True,
            token_usage={"prompt_tokens": 60, "completion_tokens": 140, "total_tokens": 200},
            model="fake-delivery-model",
        )


def create_test_client(tmp_path):
    engine = create_engine(StateStore(storage_dir=str(tmp_path)))
    engine.agents[StageType.REQUIREMENT] = FakeRequirementAgent()
    engine.agents[StageType.SOLUTION] = FakeSolutionAgent()
    engine.agents[StageType.CODING] = FakeCodeAgent()
    engine.agents[StageType.TESTING] = FakeTestAgent()
    engine.agents[StageType.REVIEW] = FakeReviewAgent()
    engine.agents[StageType.DELIVERY] = FakeDeliveryAgent()
    app = create_app(engine=engine)
    return TestClient(app), engine


def _load_backend_pipeline(client: TestClient, pipeline_id: str) -> dict:
    response = client.get(f"/api/v1/pipelines/{pipeline_id}")
    assert response.status_code == 200
    return response.json()


def test_healthcheck(tmp_path):
    client, _ = create_test_client(tmp_path)

    with client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_and_get_pipeline(tmp_path):
    client, _ = create_test_client(tmp_path)

    with client:
        create_response = client.post(
            "/api/v1/pipelines",
            json={
                "title": "Todo API",
                "requirement": "Build a todo app backend.",
            },
        )

        assert create_response.status_code == 201
        created_pipeline = create_response.json()["pipeline"]
        assert created_pipeline["title"] == "Todo API"
        assert created_pipeline["status"] == PipelineStatus.PENDING.value
        assert len(created_pipeline["stages"]) == 6

        get_response = client.get(f"/api/v1/pipelines/{created_pipeline['id']}")

    assert get_response.status_code == 200
    assert get_response.json()["id"] == created_pipeline["id"]


def test_run_pipeline_endpoint_schedules_background_work(tmp_path):
    client, engine = create_test_client(tmp_path)
    called = {"value": False}

    async def fake_run_pipeline(pipeline):
        called["value"] = True
        pipeline.status = PipelineStatus.RUNNING
        await engine.state_store.save(pipeline)

    engine.run_pipeline = fake_run_pipeline

    with client:
        create_response = client.post(
            "/api/v1/pipelines",
            json={
                "title": "Async run",
                "requirement": "Run the pipeline asynchronously.",
            },
        )
        pipeline_id = create_response.json()["pipeline"]["id"]

        run_response = client.post(f"/api/v1/pipelines/{pipeline_id}/run")
        assert run_response.status_code == 200

    assert called["value"] is True


def test_frontend_mock_dashboard_endpoints(tmp_path):
    client, _ = create_test_client(tmp_path)

    with client:
        seed_response = client.post("/api/dev/seed", params={"reset": True})
        pipelines_response = client.get("/api/pipelines")
        agents_response = client.get("/api/agents")
        checkpoints_response = client.get("/api/checkpoints", params={"status": "all"})
        analytics_response = client.get("/api/analytics")
        activities_response = client.get("/api/activities/recent")
        detail_response = client.get("/api/pipelines/pl-001")
        logs_response = client.get("/api/pipelines/pl-001/logs")

    assert seed_response.status_code == 201

    assert pipelines_response.status_code == 200
    assert len(pipelines_response.json()) == 3
    statuses = {item["status"] for item in pipelines_response.json()}
    assert "paused" in statuses
    assert "running" in statuses

    assert agents_response.status_code == 200
    assert len(agents_response.json()) == 6
    assert sum(1 for item in agents_response.json() if item["status"] == "running") == 1

    assert checkpoints_response.status_code == 200
    assert len(checkpoints_response.json()) == 2

    assert analytics_response.status_code == 200
    assert "summary" in analytics_response.json()

    assert activities_response.status_code == 200
    assert len(activities_response.json()) >= 1

    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == "pl-001"

    assert logs_response.status_code == 200
    assert len(logs_response.json()) >= 1


def test_frontend_mock_create_pipeline(tmp_path):
    client, _ = create_test_client(tmp_path)
    project_dir = tmp_path / "example-project"
    project_dir.mkdir()
    (project_dir / "package.json").write_text('{"name":"example-project"}', encoding="utf-8")
    (project_dir / "README.md").write_text("# Example Project", encoding="utf-8")
    (project_dir / "src").mkdir()
    (project_dir / "src" / "index.ts").write_text("console.log('hello')", encoding="utf-8")

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "在现有项目中实现用户收藏功能，并补充列表页。",
            },
        )

        assert create_response.status_code == 201
        created_pipeline = create_response.json()
        assert created_pipeline["status"] == "paused"
        assert created_pipeline["template"] == "新功能开发"
        assert created_pipeline["stages"][0]["status"] == "awaiting_review"
        assert created_pipeline["stages"][0]["isCheckpoint"] is True
        assert created_pipeline["stages"][0]["tokens"] == 400
        assert "需求文档" in created_pipeline["stages"][0]["output"]
        assert "项目目录扫描" in created_pipeline["stages"][0]["output"]
        assert "package.json" in created_pipeline["stages"][0]["output"]
        assert created_pipeline["projectPath"] == str(project_dir.resolve())
        assert "关键文件" in created_pipeline["projectSummary"]
        pipeline_id = created_pipeline["id"]
        assert created_pipeline["requirementDocPath"] == str(
            (project_dir / ".flowstate" / pipeline_id / "docs" / "requirements.md").resolve()
        )

        detail_response = client.get(f"/api/pipelines/{created_pipeline['id']}")
        backend_pipeline = _load_backend_pipeline(client, created_pipeline["id"])
        logs_response = client.get(f"/api/pipelines/{created_pipeline['id']}/logs")
        checkpoints_response = client.get("/api/checkpoints", params={"status": "pending"})

    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == created_pipeline["id"]
    assert detail_response.json()["status"] == "paused"
    req_doc = project_dir / ".flowstate" / pipeline_id / "docs" / "requirements.md"
    assert req_doc.exists()
    assert "收藏功能" in req_doc.read_text(encoding="utf-8")

    assert logs_response.status_code == 200
    assert any("工作目录" in log for log in logs_response.json())
    assert any("项目目录扫描完成" in log for log in logs_response.json())
    assert any("需求分析完成" in log for log in logs_response.json())
    assert any("Token 消耗: 400" in log for log in logs_response.json())
    assert any("需求文档已写入" in log for log in logs_response.json())
    assert any("等待审批" in log for log in logs_response.json())

    assert checkpoints_response.status_code == 200
    assert any(item["pipelineId"] == created_pipeline["id"] for item in checkpoints_response.json())


def test_frontend_mock_create_pipeline_rejects_invalid_project_path(tmp_path):
    client, _ = create_test_client(tmp_path)
    invalid_path = tmp_path / "missing-project"

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(invalid_path),
                "requirement": "实现一个新的通知中心。",
            },
        )

    assert create_response.status_code == 400
    assert "项目目录不存在" in create_response.json()["detail"]


def test_create_pipeline_rejects_dirty_git_repo(tmp_path):
    client, _ = create_test_client(tmp_path)
    project_dir = tmp_path / "dirty-repo"
    project_dir.mkdir()

    os.system(f"cd {project_dir} && git init -b main >/dev/null 2>&1")
    os.system(f"cd {project_dir} && git config user.name 'FlowState Test' && git config user.email 'test@flowstate.local'")
    (project_dir / "README.md").write_text("# demo\n", encoding="utf-8")
    os.system(f"cd {project_dir} && git add README.md && git commit -m 'init' >/dev/null 2>&1")
    (project_dir / "README.md").write_text("# demo\ndirty\n", encoding="utf-8")

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "在现有仓库上继续实现功能。",
            },
        )

    assert create_response.status_code == 400
    assert "未提交变更" in create_response.json()["detail"]


def test_approve_requirement_checkpoint_advances_pipeline(tmp_path):
    client, _ = create_test_client(tmp_path)
    project_dir = tmp_path / "approve-project"
    project_dir.mkdir()

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "实现一个自律应用的需求分析。",
            },
        )
        created_pipeline = create_response.json()
        checkpoint_id = f"cp-{created_pipeline['id']}-requirement_analysis"

        approve_response = client.post(f"/api/checkpoints/{checkpoint_id}/approve")
        detail_response = client.get(f"/api/pipelines/{created_pipeline['id']}")
        backend_pipeline = _load_backend_pipeline(client, created_pipeline["id"])
        pending_checkpoints_response = client.get("/api/checkpoints", params={"status": "pending"})

    assert approve_response.status_code == 200
    assert approve_response.json()["status"] == "approved"

    assert detail_response.status_code == 200
    updated_pipeline = detail_response.json()
    assert updated_pipeline["status"] == "paused"
    assert updated_pipeline["stages"][0]["status"] == "completed"
    assert updated_pipeline["stages"][1]["status"] == "awaiting_review"
    assert updated_pipeline["stages"][1]["tokens"] == 600
    assert "技术方案文档" in updated_pipeline["stages"][1]["output"]
    pipeline_id = created_pipeline["id"]
    solution_doc = project_dir / ".flowstate" / pipeline_id / "docs" / "solution.md"
    assert solution_doc.exists()
    assert "创建计划" in solution_doc.read_text(encoding="utf-8")

    assert pending_checkpoints_response.status_code == 200
    assert all(item["id"] != checkpoint_id for item in pending_checkpoints_response.json())
    assert any(item["pipelineId"] == created_pipeline["id"] and item["stage"] == "方案设计" for item in pending_checkpoints_response.json())


def test_reject_requirement_checkpoint_marks_feedback(tmp_path):
    client, _ = create_test_client(tmp_path)
    project_dir = tmp_path / "reject-project"
    project_dir.mkdir()

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "实现一个自律应用的需求分析。",
            },
        )
        created_pipeline = create_response.json()
        checkpoint_id = f"cp-{created_pipeline['id']}-requirement_analysis"

        reject_response = client.post(
            f"/api/checkpoints/{checkpoint_id}/reject",
            json={"reason": "请补充更明确的提醒策略和数据指标定义。"},
        )
        detail_response = client.get(f"/api/pipelines/{created_pipeline['id']}")

    assert reject_response.status_code == 200
    assert reject_response.json()["status"] == "rejected"
    assert "提醒策略" in (reject_response.json()["rejectReason"] or "")

    assert detail_response.status_code == 200
    updated_pipeline = detail_response.json()
    assert updated_pipeline["status"] in {"paused", "running"}
    assert updated_pipeline["stages"][0]["status"] in {"rejected", "awaiting_review"}


def test_reject_stage_resets_git_and_cleans_untracked_files(tmp_path):
    client, _ = create_test_client(tmp_path)
    service = client.app.state.pipeline_service
    project_dir = tmp_path / "reject-clean-project"
    project_dir.mkdir()

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "实现一个自律应用的需求分析。",
            },
        )
        created_pipeline = create_response.json()
        checkpoint_id = f"cp-{created_pipeline['id']}-requirement_analysis"
        client.post(f"/api/checkpoints/{checkpoint_id}/approve")

    pipeline = asyncio.run(service.get_pipeline(created_pipeline["id"]))
    assert pipeline is not None
    worktree = Path(pipeline.context.git.worktree_path or "")
    scratch = worktree / ".flowstate" / created_pipeline["id"] / "scratch.tmp"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text("stale scratch", encoding="utf-8")
    assert scratch.exists()
    solution_doc_path = Path(pipeline.context.solution_doc_path or "")
    assert solution_doc_path.exists()

    asyncio.run(service.reject_stage(created_pipeline["id"], 1, "请补充架构边界"))
    assert not scratch.exists()
    assert not solution_doc_path.exists()


def test_approve_solution_checkpoint_generates_code(tmp_path):
    client, _ = create_test_client(tmp_path)
    project_dir = tmp_path / "coding-project"
    project_dir.mkdir()

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "实现一个自律应用的需求分析。",
            },
        )
        created_pipeline = create_response.json()

        requirement_checkpoint_id = f"cp-{created_pipeline['id']}-requirement_analysis"
        client.post(f"/api/checkpoints/{requirement_checkpoint_id}/approve")

        solution_checkpoint_id = f"cp-{created_pipeline['id']}-solution_design"
        approve_solution_response = client.post(f"/api/checkpoints/{solution_checkpoint_id}/approve")
        detail_response = client.get(f"/api/pipelines/{created_pipeline['id']}")
        backend_pipeline = _load_backend_pipeline(client, created_pipeline["id"])
        checkpoints_response = client.get("/api/checkpoints", params={"status": "pending"})

    assert approve_solution_response.status_code == 200
    assert approve_solution_response.json()["status"] == "approved"

    assert detail_response.status_code == 200
    updated_pipeline = detail_response.json()
    assert updated_pipeline["status"] == "paused"
    assert updated_pipeline["stages"][1]["status"] == "completed"
    assert updated_pipeline["stages"][2]["status"] == "completed"
    assert updated_pipeline["stages"][3]["status"] == "completed"
    assert updated_pipeline["stages"][4]["status"] == "awaiting_review"
    assert updated_pipeline["stages"][2]["tokens"] == 1000
    assert updated_pipeline["stages"][3]["tokens"] == 200
    assert updated_pipeline["stages"][4]["tokens"] == 200
    assert "FastAPI()" in (updated_pipeline["stages"][2]["output"] or "")
    assert "测试报告" in (updated_pipeline["stages"][3]["output"] or "")
    assert "代码评审报告" in (updated_pipeline["stages"][4]["output"] or "")
    pipeline_id = created_pipeline["id"]
    wt = project_dir / ".flowstate" / "worktrees" / pipeline_id
    assert (wt / "app" / "main.py").exists()
    assert (wt / "requirements.txt").exists()
    assert (wt / "tests" / "test_app.py").exists()
    assert (project_dir / ".flowstate" / pipeline_id / "docs" / "test_report.md").exists()
    assert (project_dir / ".flowstate" / pipeline_id / "docs" / "review_report.md").exists()
    assert any(
        item["pipelineId"] == created_pipeline["id"] and item["stage"] == "代码评审"
        for item in checkpoints_response.json()
    )


def test_failing_test_stage_waits_for_approval_and_blocks_review(tmp_path):
    client, engine = create_test_client(tmp_path)
    engine.agents[StageType.TESTING] = FakeFailingTestAgent()
    project_dir = tmp_path / "failing-test-project"
    project_dir.mkdir()

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "实现一个自律应用并补齐测试。",
            },
        )
        created_pipeline = create_response.json()

        requirement_checkpoint_id = f"cp-{created_pipeline['id']}-requirement_analysis"
        client.post(f"/api/checkpoints/{requirement_checkpoint_id}/approve")

        solution_checkpoint_id = f"cp-{created_pipeline['id']}-solution_design"
        client.post(f"/api/checkpoints/{solution_checkpoint_id}/approve")

        detail_response = client.get(f"/api/pipelines/{created_pipeline['id']}")
        backend_pipeline = _load_backend_pipeline(client, created_pipeline["id"])
        checkpoints_response = client.get("/api/checkpoints", params={"status": "pending"})

    assert detail_response.status_code == 200
    updated_pipeline = detail_response.json()
    assert updated_pipeline["status"] == "paused"
    assert updated_pipeline["stages"][2]["status"] == "completed"
    assert updated_pipeline["stages"][3]["status"] == "awaiting_review"
    assert updated_pipeline["stages"][4]["status"] == "idle"
    assert updated_pipeline["stages"][3]["tokens"] == 180
    assert "测试报告" in (updated_pipeline["stages"][3]["output"] or "")
    pipeline_id = created_pipeline["id"]
    wt = project_dir / ".flowstate" / "worktrees" / pipeline_id
    assert (wt / "tests" / "test_app.py").exists()
    assert (project_dir / ".flowstate" / pipeline_id / "docs" / "test_report.md").exists()
    assert any(
        item["pipelineId"] == created_pipeline["id"] and item["stage"] == "测试生成"
        for item in checkpoints_response.json()
    )


def test_approve_review_checkpoint_moves_delivery_to_waiting_review(tmp_path):
    client, _ = create_test_client(tmp_path)
    project_dir = tmp_path / "delivery-project"
    project_dir.mkdir()

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "实现一个自律应用并完成交付。",
            },
        )
        created_pipeline = create_response.json()

        requirement_checkpoint_id = f"cp-{created_pipeline['id']}-requirement_analysis"
        client.post(f"/api/checkpoints/{requirement_checkpoint_id}/approve")

        solution_checkpoint_id = f"cp-{created_pipeline['id']}-solution_design"
        client.post(f"/api/checkpoints/{solution_checkpoint_id}/approve")

        review_checkpoint_id = f"cp-{created_pipeline['id']}-code_review"
        approve_review_response = client.post(f"/api/checkpoints/{review_checkpoint_id}/approve")
        detail_response = client.get(f"/api/pipelines/{created_pipeline['id']}")
        backend_pipeline = _load_backend_pipeline(client, created_pipeline["id"])
        checkpoints_response = client.get("/api/checkpoints", params={"status": "pending"})

    assert approve_review_response.status_code == 200
    assert approve_review_response.json()["status"] == "approved"

    assert detail_response.status_code == 200
    updated_pipeline = detail_response.json()
    assert updated_pipeline["status"] == "paused"
    assert updated_pipeline["stages"][4]["status"] == "completed"
    assert updated_pipeline["stages"][5]["status"] == "awaiting_review"
    assert "交付汇总" in (updated_pipeline["stages"][5]["output"] or "")
    pipeline_id = created_pipeline["id"]
    assert (project_dir / ".flowstate" / pipeline_id / "docs" / "delivery.md").exists()
    assert any(
        item["pipelineId"] == created_pipeline["id"] and item["stage"] == "交付集成"
        for item in checkpoints_response.json()
    )


def test_approve_delivery_checkpoint_completes_pipeline(tmp_path):
    client, _ = create_test_client(tmp_path)
    project_dir = tmp_path / "delivery-approve-project"
    project_dir.mkdir()

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "实现一个自律应用并完成交付。",
            },
        )
        created_pipeline = create_response.json()

        requirement_checkpoint_id = f"cp-{created_pipeline['id']}-requirement_analysis"
        client.post(f"/api/checkpoints/{requirement_checkpoint_id}/approve")

        solution_checkpoint_id = f"cp-{created_pipeline['id']}-solution_design"
        client.post(f"/api/checkpoints/{solution_checkpoint_id}/approve")

        review_checkpoint_id = f"cp-{created_pipeline['id']}-code_review"
        client.post(f"/api/checkpoints/{review_checkpoint_id}/approve")

        delivery_checkpoint_id = f"cp-{created_pipeline['id']}-delivery"
        approve_delivery_response = client.post(f"/api/checkpoints/{delivery_checkpoint_id}/approve")
        detail_response = client.get(f"/api/pipelines/{created_pipeline['id']}")

    assert approve_delivery_response.status_code == 200
    assert approve_delivery_response.json()["status"] == "approved"

    assert detail_response.status_code == 200
    updated_pipeline = detail_response.json()
    assert updated_pipeline["status"] == "completed"
    assert updated_pipeline["stages"][5]["status"] == "completed"
    assert updated_pipeline["stages"][5]["tokens"] == 200


def test_git_status_and_cleanup_endpoints(tmp_path):
    client, _ = create_test_client(tmp_path)
    project_dir = tmp_path / "git-status-project"
    project_dir.mkdir()
    (project_dir / "README.md").write_text("# demo", encoding="utf-8")

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "实现一个简单功能并写需求文档。",
            },
        )
        assert create_response.status_code == 201
        pipeline_id = create_response.json()["id"]

        status_response = client.get(f"/api/v1/pipelines/{pipeline_id}/git/status")
        diff_response = client.get(f"/api/v1/pipelines/{pipeline_id}/git/diff")
        cleanup_response = client.delete(f"/api/v1/pipelines/{pipeline_id}/git")
        status_after_cleanup = client.get(f"/api/v1/pipelines/{pipeline_id}/git/status")

    assert status_response.status_code == 200
    git_status = status_response.json()
    assert git_status["enabled"] is True
    assert git_status["mode"] == "worktree"
    assert git_status["worktree_path"]

    assert diff_response.status_code == 200

    assert cleanup_response.status_code == 200
    assert cleanup_response.json()["enabled"] is False

    assert status_after_cleanup.status_code == 200
    assert status_after_cleanup.json()["enabled"] is False


def test_git_cleanup_failure_preserves_context(tmp_path):
    client, _ = create_test_client(tmp_path)
    service = client.app.state.pipeline_service
    project_dir = tmp_path / "git-cleanup-failure-project"
    project_dir.mkdir()

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "实现一个简单功能并写需求文档。",
            },
        )
        pipeline_id = create_response.json()["id"]

    original_remove = service.git_service.remove_worktree

    def failing_remove(repo, worktree_path, *, force=False):
        raise GitError("remove failed")

    service.git_service.remove_worktree = failing_remove
    try:
        pipeline = asyncio.run(service.cleanup_pipeline_git(pipeline_id))
    finally:
        service.git_service.remove_worktree = original_remove

    assert pipeline.context.git.enabled is True
    assert pipeline.context.git.worktree_path is not None
    assert pipeline.context.git.working_branch is None or isinstance(pipeline.context.git.working_branch, str)
    assert any("保留当前 Git 上下文" in log for log in pipeline.logs)


def test_stage_commit_files_are_incremental(tmp_path):
    client, _ = create_test_client(tmp_path)
    project_dir = tmp_path / "incremental-commit-project"
    project_dir.mkdir()

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "实现一个自律应用的需求分析。",
            },
        )
        created_pipeline = create_response.json()
        requirement_checkpoint_id = f"cp-{created_pipeline['id']}-requirement_analysis"
        client.post(f"/api/checkpoints/{requirement_checkpoint_id}/approve")
        solution_checkpoint_id = f"cp-{created_pipeline['id']}-solution_design"
        client.post(f"/api/checkpoints/{solution_checkpoint_id}/approve")
        backend_pipeline = _load_backend_pipeline(client, created_pipeline["id"])

    stage_commits = backend_pipeline["context"]["git"]["stage_commits"]
    testing_commit = next(item for item in stage_commits if item["stage_type"] == "testing")
    assert "tests/test_app.py" in testing_commit["files_changed"]
    assert "app/main.py" not in testing_commit["files_changed"]


def test_dynamic_token_analytics_and_agents(tmp_path):
    client, _ = create_test_client(tmp_path)
    project_dir = tmp_path / "analytics-project"
    project_dir.mkdir()

    with client:
        create_response = client.post(
            "/api/pipelines",
            json={
                "projectPath": str(project_dir),
                "requirement": "实现一个自律应用并推进到代码评审。",
            },
        )
        created_pipeline = create_response.json()

        requirement_checkpoint_id = f"cp-{created_pipeline['id']}-requirement_analysis"
        client.post(f"/api/checkpoints/{requirement_checkpoint_id}/approve")

        solution_checkpoint_id = f"cp-{created_pipeline['id']}-solution_design"
        client.post(f"/api/checkpoints/{solution_checkpoint_id}/approve")

        analytics_response = client.get("/api/analytics")
        agents_response = client.get("/api/agents")

    assert analytics_response.status_code == 200
    analytics = analytics_response.json()
    assert analytics["summary"]["totalRuns"] == 1
    assert analytics["summary"]["totalTokens"] == 2400
    assert any(item["tokens"] > 0 for item in analytics["tokenUsage"])
    assert any(item["stage"] == "代码生成" and item["avg"] >= 0 for item in analytics["stageDurations"])

    assert agents_response.status_code == 200
    agents = agents_response.json()
    requirement_agent = next(item for item in agents if item["name"] == "RequirementsAgent")
    code_agent = next(item for item in agents if item["name"] == "CodegenAgent")
    review_agent = next(item for item in agents if item["name"] == "ReviewAgent")
    assert requirement_agent["avgTokens"] == 400
    assert code_agent["avgTokens"] == 1000
    assert review_agent["avgTokens"] == 200


def test_settings_round_trip(tmp_path):
    env_file = tmp_path / ".env"
    settings_file = tmp_path / "settings.json"
    previous_env_path = os.environ.get("FS_ENV_FILE_PATH")
    previous_settings_path = os.environ.get("FS_SETTINGS_PATH")
    os.environ["FS_ENV_FILE_PATH"] = str(env_file)
    os.environ["FS_SETTINGS_PATH"] = str(settings_file)
    client, _ = create_test_client(tmp_path)
    try:
        with client:
            get_response = client.get("/api/settings")
            assert get_response.status_code == 200
            initial = get_response.json()
            assert "llm" in initial
            assert "agentConfigs" in initial

            update_response = client.put(
                "/api/settings",
                json={
                    "llm": {
                        "provider": "deepseek",
                        "model": "deepseek-chat",
                        "baseUrl": "https://api.deepseek.com/",
                        "apiKey": "sk-test-deepseek-1234",
                    },
                    "agentConfigs": {
                        "architect-agent": {
                            "provider": "kimi",
                            "model": "kimi-k2.6",
                            "baseUrl": "https://api.moonshot.cn/v1",
                            "apiKey": "sk-agent-kimi-5678",
                        }
                    },
                    "pipeline": {
                        "defaultProvider": "deepseek",
                        "maxAgentRetries": 5,
                        "checkpointTimeoutMinutes": 45,
                        "autoCreateBranch": False,
                        "autoCommitCode": True,
                        "autoCreateMR": False,
                        "branchNamePattern": "feature/{pipeline-id}",
                        "repositoryPath": "./repo",
                        "semanticIndex": True,
                    },
                    "general": {
                        "checkpointNotifications": False,
                        "pipelineCompleteNotifications": True,
                        "agentFailureAlerts": False,
                        "logRetentionDays": "30",
                        "anonymousUsageStats": True,
                    },
                },
            )
            assert update_response.status_code == 200
            updated = update_response.json()
    finally:
        if previous_env_path is None:
            os.environ.pop("FS_ENV_FILE_PATH", None)
        else:
            os.environ["FS_ENV_FILE_PATH"] = previous_env_path
        if previous_settings_path is None:
            os.environ.pop("FS_SETTINGS_PATH", None)
        else:
            os.environ["FS_SETTINGS_PATH"] = previous_settings_path

    assert updated["llm"]["provider"] == "deepseek"
    assert updated["llm"]["model"] == "deepseek-chat"
    assert updated["llm"]["baseUrl"] == "https://api.deepseek.com/"
    assert updated["llm"]["apiKey"] == "sk-test-deepseek-1234"
    assert updated["agentConfigs"]["architect-agent"]["provider"] == "kimi"
    assert updated["agentConfigs"]["architect-agent"]["model"] == "kimi-k2.6"
    assert updated["agentConfigs"]["architect-agent"]["baseUrl"] == "https://api.moonshot.cn/v1"
    assert updated["agentConfigs"]["architect-agent"]["apiKey"] == "sk-agent-kimi-5678"
    assert updated["pipeline"]["maxAgentRetries"] == 5
    assert updated["pipeline"]["repositoryPath"] == "./repo"
    assert updated["general"]["logRetentionDays"] == "30"
    assert env_file.exists()
    env_text = env_file.read_text(encoding="utf-8")
    assert "FS_LLM_PROVIDER=deepseek" in env_text
    assert "FS_AGENT_ARCHITECT_PROVIDER=kimi" in env_text
    assert "FS_AGENT_ARCHITECT_MODEL=kimi-k2.6" in env_text
    assert "FS_AGENT_REQUIREMENTS_PROVIDER=" in env_text


def test_validate_llm_settings_rejects_invalid_format(tmp_path):
    client, _ = create_test_client(tmp_path)

    with client:
        response = client.post(
            "/api/settings/validate-llm",
            json={
                "agentId": "architect-agent",
                "llm": {
                    "provider": "unknown-provider",
                    "model": "",
                    "baseUrl": "not-a-url",
                    "apiKey": "bad key",
                },
            },
        )

    assert response.status_code == 400
    assert "Provider" in response.json()["detail"] or "URL" in response.json()["detail"]


def test_validate_llm_settings_connectivity_success(tmp_path):
    client, _ = create_test_client(tmp_path)

    async def fake_connectivity(payload):
        assert payload["provider"] == "kimi"
        assert payload["model"] == "kimi-k2.6"

    client.app.state.settings_service._test_llm_connectivity = fake_connectivity

    with client:
        response = client.post(
            "/api/settings/validate-llm",
            json={
                "agentId": "architect-agent",
                "llm": {
                    "provider": "kimi",
                    "model": "kimi-k2.6",
                    "baseUrl": "https://api.moonshot.cn/v1",
                    "apiKey": "sk-test-kimi-1234",
                },
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["provider"] == "kimi"
    assert payload["model"] == "kimi-k2.6"


def test_pipeline_pause_resume_and_cancel_actions(tmp_path):
    client, _ = create_test_client(tmp_path)
    service = client.app.state.pipeline_service

    pipeline = asyncio.run(
        service.create_pipeline(
            title="Manual control",
            requirement="manually control pipeline",
            start_immediately=False,
        )
    )
    pipeline.status = PipelineStatus.RUNNING
    pipeline.stages[0].status = StageStatus.RUNNING
    asyncio.run(service.state_store.save(pipeline))

    with client:
        pause_response = client.post(f"/api/pipelines/{pipeline.id}/pause")
        resume_response = client.post(f"/api/pipelines/{pipeline.id}/resume")
        cancel_response = client.post(f"/api/pipelines/{pipeline.id}/cancel")

    assert pause_response.status_code == 200
    assert pause_response.json()["status"] == "paused"
    assert resume_response.status_code == 200
    assert resume_response.json()["status"] == "running"
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"


def test_ui_api_routes_smoke(tmp_path):
    client, _ = create_test_client(tmp_path)

    with client:
        seed_response = client.post("/api/dev/seed", params={"reset": True})
        pipelines_response = client.get("/api/pipelines")
        agents_response = client.get("/api/agents")
        checkpoints_response = client.get("/api/checkpoints", params={"status": "all"})
        analytics_response = client.get("/api/analytics")
        activities_response = client.get("/api/activities/recent")

        pipeline_id = pipelines_response.json()[0]["id"] if pipelines_response.json() else ""
        detail_response = client.get(f"/api/pipelines/{pipeline_id}") if pipeline_id else None
        logs_response = client.get(f"/api/pipelines/{pipeline_id}/logs") if pipeline_id else None

    assert seed_response.status_code == 201
    assert pipelines_response.status_code == 200
    assert agents_response.status_code == 200
    assert checkpoints_response.status_code == 200
    assert analytics_response.status_code == 200
    assert activities_response.status_code == 200
    assert pipeline_id
    assert detail_response is not None and detail_response.status_code == 200
    assert logs_response is not None and logs_response.status_code == 200


def test_requirement_agent_renders_markdown_document_from_structured_json():
    agent = RequirementAgent()

    async def fake_call_llm_response(*args, **kwargs):
        return LLMResponse(content="""{
  "title": "收藏功能需求分析",
  "business_goals": ["提升用户留存", "支持用户管理感兴趣内容"],
  "functional_requirements": ["列表页展示收藏状态", "支持收藏与取消收藏"],
  "user_stories": ["作为用户，我希望在列表页快速收藏内容"],
  "acceptance_criteria": ["用户可以成功收藏和取消收藏目标内容"],
  "non_functional_requirements": ["接口响应时间应控制在 300ms 内"],
  "open_questions": ["是否需要支持批量收藏"],
  "modules": [
    {
      "module_name": "收藏状态展示",
      "priority": "P0",
      "description": "在列表页展示已收藏和未收藏两种状态。"
    }
  ]
}""", usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, model="fake-requirement-model")

    agent.call_llm_response = fake_call_llm_response
    output = asyncio.run(
        agent.execute(
            AgentInput(
                task_description="执行阶段: requirement_analysis",
                context={
                    "requirement_raw": "实现收藏与取消收藏能力",
                    "project_path": "/tmp/demo-project",
                    "project_summary": "## 项目目录扫描\n\n- 关键文件：package.json",
                },
            )
        )
    )

    assert output.success is True
    assert output.result["modules"][0]["module_name"] == "收藏状态展示"
    assert "# 需求文档" in output.result["document"]
    assert "## 功能清单" in output.result["document"]
    assert "package.json" in output.result["document"]
    assert "```json" in output.result["document"]


def test_solution_agent_renders_concise_solution_document():
    from src.agents.solution_agent import SolutionAgent

    agent = SolutionAgent()

    async def fake_call_llm_response(*args, **kwargs):
        return LLMResponse(content="""{
  "title": "自律 App 技术方案",
  "architecture_overview": "前后端分离，移动端配合后端 API 提供能力。",
  "tech_stack": ["React Native", "FastAPI", "PostgreSQL"],
  "directory_structure": [
    {"path": "mobile/", "purpose": "移动端应用"},
    {"path": "backend/", "purpose": "服务端接口"}
  ],
  "api_design": [
    {
      "name": "创建计划",
      "method": "POST",
      "path": "/plans",
      "description": "创建每日计划",
      "request": "标题、日期、优先级",
      "response": "计划详情"
    }
  ],
  "data_models": [
    {
      "name": "Plan",
      "description": "每日计划实体",
      "fields": [
        {"name": "id", "type": "uuid", "required": true, "description": "计划 ID"}
      ]
    }
  ],
  "technical_rationale": ["React Native 适合跨平台快速交付"],
  "risks": ["提醒功能依赖系统通知权限"],
  "open_questions": ["是否支持 Apple Watch 通知"]
}""", usage={"prompt_tokens": 20, "completion_tokens": 40, "total_tokens": 60}, model="fake-solution-model")

    requirement_doc = """# 需求文档

## 标题
自律 App

## 原始需求
写一个自律 app

## 业务目标
- 提升计划执行率

## 功能清单
- 每日计划
- 专注计时

## 验收标准
- 能创建计划

## 模块划分
### 每日计划（P0）
计划管理
"""

    agent.call_llm_response = fake_call_llm_response
    output = asyncio.run(
        agent.execute(
            AgentInput(
                task_description="执行阶段: solution_design",
                context={
                    "requirement_doc": requirement_doc,
                    "project_path": "/tmp/demo-project",
                    "project_summary": "## 项目目录扫描\n\n- 关键文件：package.json",
                },
            )
        )
    )

    assert output.success is True
    assert "# 技术方案文档" in output.result["design"]
    assert "## 需求摘要" in output.result["design"]
    assert "## 原始需求" not in output.result["design"]
    assert "## Stage3 使用的技术栈约束" in output.result["design"]
    assert "## Stage3 文件清单" in output.result["design"]
    assert "创建计划" in output.result["design"]


def test_code_agent_requires_file_plan_and_returns_multiple_files():
    from src.agents.code_agent import CodeAgent

    agent = CodeAgent()

    async def fake_call_llm_response(*args, **kwargs):
        main_py = base64.b64encode(
            b"from fastapi import FastAPI\n\napp = FastAPI()\n"
        ).decode("utf-8")
        requirements_txt = base64.b64encode(
            b"fastapi\nuvicorn\n"
        ).decode("utf-8")
        return LLMResponse(content="""<file path="app/main.py">
<summary>FastAPI 入口</summary>
<content_base64>
__MAIN_PY__
</content_base64>
</file>
<file path="requirements.txt">
<summary>依赖定义</summary>
<content_base64>
__REQ_TXT__
</content_base64>
</file>""".replace("__MAIN_PY__", main_py).replace("__REQ_TXT__", requirements_txt), usage={"prompt_tokens": 30, "completion_tokens": 70, "total_tokens": 100}, model="fake-code-model")

    agent.call_llm_response = fake_call_llm_response
    output = asyncio.run(
        agent.execute(
            AgentInput(
                task_description="执行阶段: coding",
                context={
                    "requirement_doc": "# 需求文档",
                    "solution_doc": "# 技术方案文档",
                    "solution_structured": {
                        "resolved_stack": {
                            "frontend": "React Native",
                            "backend": "FastAPI",
                            "database": "SQLite",
                            "state_management": "Zustand",
                            "notifications": "Expo Notifications",
                            "testing": "pytest + vitest",
                        },
                        "file_plan": [
                            {
                                "path": "app/main.py",
                                "layer": "backend",
                                "purpose": "FastAPI 入口",
                                "must_generate": True,
                            },
                            {
                                "path": "requirements.txt",
                                "layer": "config",
                                "purpose": "依赖定义",
                                "must_generate": True,
                            },
                        ],
                    },
                },
            )
        )
    )

    assert output.success is True
    assert set(output.result["files"].keys()) == {"app/main.py", "requirements.txt"}


def test_code_agent_generates_required_files_in_batches():
    from src.agents.code_agent import CodeAgent

    agent = CodeAgent()
    agent.BATCH_SIZE = 2
    call_count = {"value": 0}

    async def fake_call_llm_response(message: str, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            main_py = base64.b64encode(
                b"from fastapi import FastAPI\n\napp = FastAPI()\n"
            ).decode("utf-8")
            return LLMResponse(content=f"""<file path="app/main.py">
<summary>入口</summary>
<content_base64>
{main_py}
</content_base64>
</file>
""", usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, model="fake-code-model")

        if call_count["value"] == 2:
            router_py = base64.b64encode(
                b"from fastapi import APIRouter\n\nrouter = APIRouter()\n"
            ).decode("utf-8")
            return LLMResponse(content=f"""<file path="app/router.py">
<summary>路由</summary>
<content_base64>
{router_py}
</content_base64>
</file>""", usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, model="fake-code-model")

        requirements_txt = base64.b64encode(
            b"fastapi\nuvicorn\n"
        ).decode("utf-8")
        return LLMResponse(content=f"""<file path="requirements.txt">
<summary>依赖</summary>
<content_base64>
{requirements_txt}
</content_base64>
</file>""", usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, model="fake-code-model")

    agent.call_llm_response = fake_call_llm_response
    output = asyncio.run(
        agent.execute(
            AgentInput(
                task_description="执行阶段: coding",
                context={
                    "requirement_doc": "# 需求文档",
                    "solution_doc": "# 技术方案文档",
                    "solution_structured": {
                        "resolved_stack": {
                            "frontend": "React Native",
                            "backend": "FastAPI",
                            "database": "SQLite",
                            "state_management": "Zustand",
                            "notifications": "Expo Notifications",
                            "testing": "pytest + vitest",
                        },
                        "file_plan": [
                            {"path": "app/main.py", "layer": "backend", "purpose": "入口", "must_generate": True},
                            {"path": "app/router.py", "layer": "backend", "purpose": "路由", "must_generate": True},
                            {"path": "requirements.txt", "layer": "config", "purpose": "依赖", "must_generate": True},
                        ],
                    },
                },
            )
        )
    )

    assert output.success is True
    # 3 files (1 call each) + 1 consistency review call = 4
    assert call_count["value"] == 4
    assert set(output.result["files"].keys()) == {"app/main.py", "app/router.py", "requirements.txt"}
