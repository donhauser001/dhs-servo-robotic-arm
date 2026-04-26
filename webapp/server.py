"""FastAPI 后端：把 :class:`webapp.state.ArmService` 暴露成 REST + WebSocket。

启动：
    arm-web                                     # 默认 0.0.0.0:8000
    arm-web --host 127.0.0.1 --port 9000        # 自定义
    python -m webapp.server --config foo.yaml   # 自定义 yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .state import ArmService

log = logging.getLogger("webapp")

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_CONFIG = ROOT / "config" / "arm_config.yaml"


# ---------------------------------------------------------------------------
# 请求体模型
# ---------------------------------------------------------------------------

class DisableBody(BaseModel):
    to: Optional[str] = "safe"


class TorqueBody(BaseModel):
    on: bool
    joint: Optional[str] = None


class MoveBody(BaseModel):
    pcts: Optional[List[float]] = None
    degs: Optional[List[float]] = None
    raws: Optional[List[int]] = None
    speed: Optional[int] = None
    acc: Optional[int] = None
    wait: bool = False
    bypass_constraints: bool = False


class MoveSingleBody(BaseModel):
    pct: Optional[float] = None
    deg: Optional[float] = None
    raw: Optional[int] = None
    speed: Optional[int] = None
    acc: Optional[int] = None


class GotoPoseBody(BaseModel):
    name: str
    speed: int = 600
    acc: int = 30
    wait: bool = True


class MoveHomeBody(BaseModel):
    speed: int = 600
    acc: int = 30


class CaptureRawBody(BaseModel):
    field: str = Field(..., pattern="^(home_raw|max_raw|low_raw)$")


class CalibrationBody(BaseModel):
    """更新关节标定字段（None 表示不动）。"""
    home_raw: Optional[int] = None
    max_raw: Optional[int] = None
    low_raw: Optional[int] = None  # 显式传 null 想清空：用 unset_low
    min_deg: Optional[float] = None
    max_deg: Optional[float] = None
    home_deg: Optional[float] = None
    unset_low: bool = False
    unset_deg: bool = False


class ConstraintBody(BaseModel):
    """联锁约束：当 gate 关节不满足 ``op gate_min_deg`` 条件时 target 关节被冻结。

    ``op = ">"``：gate > gate_min_deg 才解锁；
    ``op = "<"``：gate < gate_min_deg 才解锁。
    """
    target: str
    gate: str
    gate_min_deg: float
    op: Literal[">", "<"] = ">"
    enabled: bool = True
    note: str = ""


class ConstraintUpdateBody(BaseModel):
    """改阈值或方向 / 启用状态 / 备注（target / gate 不变）。"""
    gate_min_deg: Optional[float] = None
    op: Optional[Literal[">", "<"]] = None
    enabled: Optional[bool] = None
    note: Optional[str] = None


class PrerequisiteBody(BaseModel):
    """前置联动：target 想去 ``target_op target_deg`` 时，gate 必须先满足
    ``gate_op gate_deg``；不满足则自动先把 gate 转到带 ``buffer_deg`` 余量
    的安全位置后再发原动作。"""
    target: str
    target_op: Literal[">", "<"] = ">"
    target_deg: float
    gate: str
    gate_op: Literal[">", "<"] = ">"
    gate_deg: float
    buffer_deg: float = 2.0
    enabled: bool = True
    note: str = ""


class PrerequisiteUpdateBody(BaseModel):
    """改任意字段（target / gate 不变）。"""
    target_op: Optional[Literal[">", "<"]] = None
    target_deg: Optional[float] = None
    gate_op: Optional[Literal[">", "<"]] = None
    gate_deg: Optional[float] = None
    buffer_deg: Optional[float] = None
    enabled: Optional[bool] = None
    note: Optional[str] = None


class ZoneLimitBody(BaseModel):
    """条件限位：当 gate ``gate_op gate_deg`` 时，target 禁止 ``target_op target_deg``。"""
    target: str
    target_op: Literal[">", "<"] = ">"
    target_deg: float
    gate: str
    gate_op: Literal[">", "<"] = ">"
    gate_deg: float
    enabled: bool = True
    note: str = ""


class ZoneLimitUpdateBody(BaseModel):
    target_op: Optional[Literal[">", "<"]] = None
    target_deg: Optional[float] = None
    gate_op: Optional[Literal[">", "<"]] = None
    gate_deg: Optional[float] = None
    enabled: Optional[bool] = None
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# YAML 读写（保留注释）
# ---------------------------------------------------------------------------

def _yaml():
    from ruamel.yaml import YAML

    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 120
    return y


def _load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return _yaml().load(f)


def _dump_yaml(path: Path, data) -> None:
    buf = StringIO()
    _yaml().dump(data, buf)
    with open(path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())


def _yaml_update_pose(path: Path, name: str, raws: List[int]) -> None:
    data = _load_yaml(path)
    if data.get("poses") is None:
        data["poses"] = {}
    data["poses"][name] = list(raws)
    _dump_yaml(path, data)


def _yaml_delete_pose(path: Path, name: str) -> bool:
    data = _load_yaml(path)
    poses = data.get("poses") or {}
    if name not in poses:
        return False
    del poses[name]
    _dump_yaml(path, data)
    return True


def _yaml_joint_names(data) -> List[str]:
    return [j.get("name") for j in (data.get("joints") or [])]


def _yaml_add_constraint(path: Path, body: ConstraintBody) -> Dict[str, Any]:
    """追加一条联锁约束。校验 target/gate 必须是已存在的关节，且 (target, gate) 不重复。"""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    data = _load_yaml(path)
    names = _yaml_joint_names(data)
    if body.target not in names:
        raise ValueError(f"target 不是已知关节：{body.target}")
    if body.gate not in names:
        raise ValueError(f"gate 不是已知关节：{body.gate}")
    if body.target == body.gate:
        raise ValueError("target 与 gate 不能是同一个关节")

    constraints = data.get("constraints")
    if constraints is None:
        constraints = CommentedSeq()
        data["constraints"] = constraints

    for c in constraints:
        if c.get("target") == body.target and c.get("gate") == body.gate:
            raise ValueError(
                f"约束已存在：{body.target} ← {body.gate}（请改阈值或先删旧约束）"
            )

    # ruamel 会把"列表最后一项之后的空行 + 下一段注释"挂在最后一项末尾字段的 EOL 槽里。
    # 如果直接 append，新元素会被插到这个段尾注释之后（视觉上属于下一段）。
    # 解决：在 append 之前把这种"段间 EOL 注释"摘下来，append 完再挂到新元素的末尾字段上。
    saved_section_break = None  # (last_item, last_field) 占位
    if constraints:
        last = constraints[-1]
        if isinstance(last, CommentedMap) and last:
            last_field = list(last.keys())[-1]
            slot = last.ca.items.get(last_field)
            if slot and len(slot) > 2 and slot[2] is not None:
                tok = slot[2]
                tok_value = getattr(tok, "value", "") or ""
                # 段间注释的特征：含 "\n\n"（空行 + 注释）或纯空行
                if "\n\n" in tok_value or tok_value.strip().startswith("#"):
                    saved_section_break = (last, last_field, slot[2])
                    slot[2] = None

    item = CommentedMap()
    item["target"] = body.target
    item["gate"] = body.gate
    item["gate_min_deg"] = float(body.gate_min_deg)
    item["op"] = body.op
    last_field = "op"
    if not body.enabled:
        item["enabled"] = False
        last_field = "enabled"
    if body.note:
        item["note"] = body.note
        last_field = "note"
    constraints.append(item)

    if saved_section_break is not None:
        _, _, tok = saved_section_break
        new_slot = item.ca.items.setdefault(last_field, [None, None, None, None])
        new_slot[2] = tok

    _dump_yaml(path, data)
    return dict(item)


def _yaml_update_constraint(
    path: Path, target: str, gate: str, body: "ConstraintUpdateBody",
) -> Dict[str, Any]:
    if all(getattr(body, f) is None for f in (
        "gate_min_deg", "op", "enabled", "note",
    )):
        raise ValueError("update 至少要传一个字段")
    data = _load_yaml(path)
    constraints = data.get("constraints") or []
    for c in constraints:
        if c.get("target") == target and c.get("gate") == gate:
            if body.gate_min_deg is not None:
                c["gate_min_deg"] = float(body.gate_min_deg)
            if body.op is not None:
                c["op"] = body.op
            if body.enabled is not None:
                c["enabled"] = bool(body.enabled)
            if body.note is not None:
                c["note"] = str(body.note)
            _dump_yaml(path, data)
            return dict(c)
    raise KeyError(f"未找到约束：{target} ← {gate}")


def _yaml_delete_constraint(path: Path, target: str, gate: str) -> bool:
    from ruamel.yaml.comments import CommentedMap

    data = _load_yaml(path)
    constraints = data.get("constraints") or []
    n0 = len(constraints)
    if n0 == 0:
        return False

    # 找命中索引
    hit = -1
    for i, c in enumerate(constraints):
        if c.get("target") == target and c.get("gate") == gate:
            hit = i
            break
    if hit < 0:
        return False

    # 如果删的是最后一项，要把它末尾字段挂着的"段尾注释（# 下一段...）"
    # 挪回到新最后一项的同等位置，避免段间注释跟着被删元素一起没掉。
    rescued_section_break = None
    if hit == len(constraints) - 1:
        last = constraints[hit]
        if isinstance(last, CommentedMap) and last:
            last_field = list(last.keys())[-1]
            slot = last.ca.items.get(last_field)
            if slot and len(slot) > 2 and slot[2] is not None:
                tok_value = getattr(slot[2], "value", "") or ""
                if "\n\n" in tok_value or tok_value.strip().startswith("#"):
                    rescued_section_break = slot[2]

    del constraints[hit]

    if rescued_section_break is not None and constraints:
        new_last = constraints[-1]
        # ruamel 在 flow-style mapping（{a: 1, b: 2}）上挂 EOL 注释会
        # 把注释插到 `}` 之前破坏语法，所以只对 block-style 做注释转移。
        is_block = isinstance(new_last, CommentedMap) and new_last \
            and getattr(new_last, "fa", None) is not None \
            and not new_last.fa.flow_style()
        if is_block:
            new_last_field = list(new_last.keys())[-1]
            new_slot = new_last.ca.items.setdefault(
                new_last_field, [None, None, None, None],
            )
            new_slot[2] = rescued_section_break

    if not constraints and "constraints" in data:
        # 整段删掉，避免留 `constraints: []` 这种空段
        del data["constraints"]

    _dump_yaml(path, data)
    return True


def _yaml_add_prerequisite(path: Path, body: PrerequisiteBody) -> Dict[str, Any]:
    """追加一条 prerequisite。校验 target/gate 必须是已存在的关节，
    且 (target, gate) 不重复。同样保留段间注释（参考 _yaml_add_constraint）。"""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    data = _load_yaml(path)
    names = _yaml_joint_names(data)
    if body.target not in names:
        raise ValueError(f"target 不是已知关节：{body.target}")
    if body.gate not in names:
        raise ValueError(f"gate 不是已知关节：{body.gate}")
    if body.target == body.gate:
        raise ValueError("target 与 gate 不能是同一个关节")

    prereqs = data.get("prerequisites")
    if prereqs is None:
        prereqs = CommentedSeq()
        data["prerequisites"] = prereqs

    for p in prereqs:
        if p.get("target") == body.target and p.get("gate") == body.gate:
            raise ValueError(
                f"前置联动已存在：{body.target} ← {body.gate}（请改字段或先删旧条目）"
            )

    saved_section_break = None
    if prereqs:
        last = prereqs[-1]
        if isinstance(last, CommentedMap) and last:
            last_field = list(last.keys())[-1]
            slot = last.ca.items.get(last_field)
            if slot and len(slot) > 2 and slot[2] is not None:
                tok = slot[2]
                tok_value = getattr(tok, "value", "") or ""
                if "\n\n" in tok_value or tok_value.strip().startswith("#"):
                    saved_section_break = (last, last_field, slot[2])
                    slot[2] = None

    item = CommentedMap()
    item["target"] = body.target
    item["target_op"] = body.target_op
    item["target_deg"] = float(body.target_deg)
    item["gate"] = body.gate
    item["gate_op"] = body.gate_op
    item["gate_deg"] = float(body.gate_deg)
    item["buffer_deg"] = float(body.buffer_deg)
    last_field = "buffer_deg"
    if not body.enabled:
        item["enabled"] = False
        last_field = "enabled"
    if body.note:
        item["note"] = body.note
        last_field = "note"
    prereqs.append(item)

    if saved_section_break is not None:
        _, _, tok = saved_section_break
        new_slot = item.ca.items.setdefault(last_field, [None, None, None, None])
        new_slot[2] = tok

    _dump_yaml(path, data)
    return dict(item)


def _yaml_update_prerequisite(
    path: Path, target: str, gate: str, body: "PrerequisiteUpdateBody",
) -> Dict[str, Any]:
    if all(getattr(body, f) is None for f in (
        "target_op", "target_deg", "gate_op", "gate_deg", "buffer_deg",
        "enabled", "note",
    )):
        raise ValueError("update 至少要传一个字段")
    data = _load_yaml(path)
    prereqs = data.get("prerequisites") or []
    for p in prereqs:
        if p.get("target") == target and p.get("gate") == gate:
            if body.target_op is not None: p["target_op"] = body.target_op
            if body.target_deg is not None: p["target_deg"] = float(body.target_deg)
            if body.gate_op is not None: p["gate_op"] = body.gate_op
            if body.gate_deg is not None: p["gate_deg"] = float(body.gate_deg)
            if body.buffer_deg is not None: p["buffer_deg"] = float(body.buffer_deg)
            if body.enabled is not None: p["enabled"] = bool(body.enabled)
            if body.note is not None: p["note"] = str(body.note)
            _dump_yaml(path, data)
            return dict(p)
    raise KeyError(f"未找到前置联动：{target} ← {gate}")


def _yaml_delete_prerequisite(path: Path, target: str, gate: str) -> bool:
    from ruamel.yaml.comments import CommentedMap

    data = _load_yaml(path)
    prereqs = data.get("prerequisites") or []
    if not prereqs:
        return False

    hit = -1
    for i, p in enumerate(prereqs):
        if p.get("target") == target and p.get("gate") == gate:
            hit = i
            break
    if hit < 0:
        return False

    rescued_section_break = None
    if hit == len(prereqs) - 1:
        last = prereqs[hit]
        if isinstance(last, CommentedMap) and last:
            last_field = list(last.keys())[-1]
            slot = last.ca.items.get(last_field)
            if slot and len(slot) > 2 and slot[2] is not None:
                tok_value = getattr(slot[2], "value", "") or ""
                if "\n\n" in tok_value or tok_value.strip().startswith("#"):
                    rescued_section_break = slot[2]

    del prereqs[hit]

    if rescued_section_break is not None and prereqs:
        new_last = prereqs[-1]
        is_block = isinstance(new_last, CommentedMap) and new_last \
            and getattr(new_last, "fa", None) is not None \
            and not new_last.fa.flow_style()
        if is_block:
            new_last_field = list(new_last.keys())[-1]
            new_slot = new_last.ca.items.setdefault(
                new_last_field, [None, None, None, None],
            )
            new_slot[2] = rescued_section_break

    if not prereqs and "prerequisites" in data:
        del data["prerequisites"]

    _dump_yaml(path, data)
    return True


def _yaml_add_zone_limit(path: Path, body: ZoneLimitBody) -> Dict[str, Any]:
    """追加一条 zone_limit。"""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    data = _load_yaml(path)
    names = _yaml_joint_names(data)
    if body.target not in names:
        raise ValueError(f"target 不是已知关节：{body.target}")
    if body.gate not in names:
        raise ValueError(f"gate 不是已知关节：{body.gate}")
    if body.target == body.gate:
        raise ValueError("target 与 gate 不能是同一个关节")

    items_seq = data.get("zone_limits")
    if items_seq is None:
        items_seq = CommentedSeq()
        data["zone_limits"] = items_seq

    for z in items_seq:
        if z.get("target") == body.target and z.get("gate") == body.gate:
            raise ValueError(
                f"条件限位已存在：{body.target} ← {body.gate}（请改字段或先删旧条目）"
            )

    saved_section_break = None
    if items_seq:
        last = items_seq[-1]
        if isinstance(last, CommentedMap) and last:
            last_field = list(last.keys())[-1]
            slot = last.ca.items.get(last_field)
            if slot and len(slot) > 2 and slot[2] is not None:
                tok = slot[2]
                tok_value = getattr(tok, "value", "") or ""
                if "\n\n" in tok_value or tok_value.strip().startswith("#"):
                    saved_section_break = (last, last_field, slot[2])
                    slot[2] = None

    item = CommentedMap()
    item["target"] = body.target
    item["target_op"] = body.target_op
    item["target_deg"] = float(body.target_deg)
    item["gate"] = body.gate
    item["gate_op"] = body.gate_op
    item["gate_deg"] = float(body.gate_deg)
    last_field = "gate_deg"
    if not body.enabled:
        item["enabled"] = False
        last_field = "enabled"
    if body.note:
        item["note"] = body.note
        last_field = "note"
    items_seq.append(item)

    if saved_section_break is not None:
        _, _, tok = saved_section_break
        new_slot = item.ca.items.setdefault(last_field, [None, None, None, None])
        new_slot[2] = tok

    _dump_yaml(path, data)
    return dict(item)


def _yaml_update_zone_limit(
    path: Path, target: str, gate: str, body: "ZoneLimitUpdateBody",
) -> Dict[str, Any]:
    if all(getattr(body, f) is None for f in (
        "target_op", "target_deg", "gate_op", "gate_deg",
        "enabled", "note",
    )):
        raise ValueError("update 至少要传一个字段")
    data = _load_yaml(path)
    items_seq = data.get("zone_limits") or []
    for z in items_seq:
        if z.get("target") == target and z.get("gate") == gate:
            if body.target_op is not None: z["target_op"] = body.target_op
            if body.target_deg is not None: z["target_deg"] = float(body.target_deg)
            if body.gate_op is not None: z["gate_op"] = body.gate_op
            if body.gate_deg is not None: z["gate_deg"] = float(body.gate_deg)
            if body.enabled is not None: z["enabled"] = bool(body.enabled)
            if body.note is not None: z["note"] = str(body.note)
            _dump_yaml(path, data)
            return dict(z)
    raise KeyError(f"未找到条件限位：{target} ← {gate}")


def _yaml_delete_zone_limit(path: Path, target: str, gate: str) -> bool:
    from ruamel.yaml.comments import CommentedMap

    data = _load_yaml(path)
    items_seq = data.get("zone_limits") or []
    if not items_seq:
        return False

    hit = -1
    for i, z in enumerate(items_seq):
        if z.get("target") == target and z.get("gate") == gate:
            hit = i
            break
    if hit < 0:
        return False

    rescued_section_break = None
    if hit == len(items_seq) - 1:
        last = items_seq[hit]
        if isinstance(last, CommentedMap) and last:
            last_field = list(last.keys())[-1]
            slot = last.ca.items.get(last_field)
            if slot and len(slot) > 2 and slot[2] is not None:
                tok_value = getattr(slot[2], "value", "") or ""
                if "\n\n" in tok_value or tok_value.strip().startswith("#"):
                    rescued_section_break = slot[2]

    del items_seq[hit]

    if rescued_section_break is not None and items_seq:
        new_last = items_seq[-1]
        is_block = isinstance(new_last, CommentedMap) and new_last \
            and getattr(new_last, "fa", None) is not None \
            and not new_last.fa.flow_style()
        if is_block:
            new_last_field = list(new_last.keys())[-1]
            new_slot = new_last.ca.items.setdefault(
                new_last_field, [None, None, None, None],
            )
            new_slot[2] = rescued_section_break

    if not items_seq and "zone_limits" in data:
        del data["zone_limits"]

    _dump_yaml(path, data)
    return True


def _yaml_update_joint(path: Path, joint_name: str, body: CalibrationBody) -> Dict[str, Any]:
    """按字段更新某关节；返回最终的 joint dict。"""
    data = _load_yaml(path)
    joints = data.get("joints") or []
    target = None
    for j in joints:
        if j.get("name") == joint_name:
            target = j
            break
    if target is None:
        raise KeyError(joint_name)

    # 字段更新
    for fld in ("home_raw", "max_raw"):
        v = getattr(body, fld)
        if v is not None:
            target[fld] = int(v)
    if body.low_raw is not None:
        target["low_raw"] = int(body.low_raw)
    if body.unset_low and "low_raw" in target:
        del target["low_raw"]

    if body.unset_deg:
        for fld in ("min_deg", "max_deg", "home_deg"):
            target.pop(fld, None)
    else:
        for fld in ("min_deg", "max_deg", "home_deg"):
            v = getattr(body, fld)
            if v is not None:
                target[fld] = float(v)

    _dump_yaml(path, data)
    return dict(target)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(config_path: Path) -> FastAPI:
    service = ArmService(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.startup()
        yield
        await service.shutdown()

    app = FastAPI(title="6-DOF Arm Web Console", lifespan=lifespan)
    app.state.service = service
    app.state.config_path = config_path

    # 把规则触发的 RuntimeError（如 prereq 超时未到位）转成 503，
    # 让前端能把它当"硬件/规则相关的临时不可用"展示，而不是 500 internal。
    from fastapi import Request
    from fastapi.responses import JSONResponse

    @app.exception_handler(RuntimeError)
    async def _runtime_to_503(request: "Request", exc: RuntimeError):
        msg = str(exc)
        if "未到位" in msg or "前置联动" in msg or "ArmService" in msg:
            return JSONResponse({"detail": msg}, status_code=503)
        return JSONResponse({"detail": msg}, status_code=500)

    # ---- 静态前端 ----
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # ---- 健康 / 状态 ----

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {
            "ready": service.ready,
            "error": service.error,
            "config_path": str(config_path),
        }

    @app.get("/api/state")
    async def get_state() -> Dict[str, Any]:
        if not service.ready:
            raise HTTPException(503, service.error or "not ready")
        return await service.snapshot()

    @app.get("/api/feedback")
    async def get_feedback() -> Dict[str, Any]:
        if not service.ready:
            raise HTTPException(503, service.error or "not ready")
        return await service.feedback_full()

    # ---- 控制 ----

    @app.post("/api/enable")
    async def api_enable() -> Dict[str, str]:
        await service.enable()
        return {"state": "enabled"}

    @app.post("/api/disable")
    async def api_disable(body: DisableBody) -> Dict[str, str]:
        await service.disable(to=body.to)
        return {"state": "disabled"}

    @app.post("/api/emergency-stop")
    async def api_estop() -> Dict[str, str]:
        await service.emergency_stop()
        return {"state": "disabled"}

    @app.post("/api/torque")
    async def api_torque(body: TorqueBody) -> Dict[str, Any]:
        await service.set_torque(body.on, joint=body.joint)
        return {"on": body.on, "joint": body.joint}

    @app.post("/api/move")
    async def api_move(body: MoveBody) -> Dict[str, str]:
        if body.pcts is not None:
            await service.move_pct(body.pcts, speed=body.speed, acc=body.acc, wait=body.wait)
        elif body.degs is not None:
            await service.move_deg(body.degs, speed=body.speed, acc=body.acc, wait=body.wait)
        elif body.raws is not None:
            await service.move_raw(
                body.raws, speed=body.speed, acc=body.acc, wait=body.wait,
                bypass_constraints=body.bypass_constraints,
            )
        else:
            raise HTTPException(400, "需要传 pcts / degs / raws 之一")
        return {"status": "sent"}

    @app.post("/api/move-single/{name}")
    async def api_move_single(name: str, body: MoveSingleBody) -> Dict[str, str]:
        if body.pct is None and body.deg is None and body.raw is None:
            raise HTTPException(400, "需要传 pct / deg / raw 之一")
        try:
            await service.move_single_joint(
                name, pct=body.pct, deg=body.deg, raw=body.raw,
                speed=body.speed, acc=body.acc,
            )
        except KeyError:
            raise HTTPException(404, f"无此关节：{name}")
        return {"status": "sent"}

    @app.post("/api/goto-pose")
    async def api_goto_pose(body: GotoPoseBody) -> Dict[str, str]:
        try:
            await service.goto_pose(body.name, speed=body.speed, acc=body.acc, wait=body.wait)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"status": "sent"}

    @app.post("/api/move-home")
    async def api_move_home(
        body: MoveHomeBody = Body(default_factory=MoveHomeBody),
    ) -> Dict[str, str]:
        await service.move_home(speed=body.speed, acc=body.acc)
        return {"status": "sent"}

    # ---- 配置：姿态 ----

    @app.post("/api/poses/{name}/capture")
    async def api_capture_pose(name: str) -> Dict[str, Any]:
        """读当前 raw 一帧，存为名为 ``name`` 的 pose（写回 yaml）。"""
        snap = await service.snapshot()
        raws = [j["raw"] for j in snap["joints"]]
        await asyncio.get_running_loop().run_in_executor(
            None, _yaml_update_pose, config_path, name, raws,
        )
        await service.reload_config()
        return {"name": name, "raws": raws}

    @app.delete("/api/poses/{name}")
    async def api_delete_pose(name: str) -> Dict[str, str]:
        ok = await asyncio.get_running_loop().run_in_executor(
            None, _yaml_delete_pose, config_path, name,
        )
        if not ok:
            raise HTTPException(404, f"无此姿态：{name}")
        await service.reload_config()
        return {"deleted": name}

    # ---- 配置：标定 ----

    @app.put("/api/joints/{name}/calibration")
    async def api_calibrate(name: str, body: CalibrationBody) -> Dict[str, Any]:
        try:
            updated = await asyncio.get_running_loop().run_in_executor(
                None, _yaml_update_joint, config_path, name, body,
            )
        except KeyError:
            raise HTTPException(404, f"无此关节：{name}")
        await service.reload_config()
        return {"joint": name, "fields": updated}

    @app.post("/api/joints/{name}/capture-raw")
    async def api_capture_raw(name: str, body: CaptureRawBody) -> Dict[str, Any]:
        """读当前 raw 写到该关节的 ``home_raw`` / ``max_raw`` / ``low_raw`` 之一。"""
        try:
            cur = await service.capture_joint_value(name, body.field)
        except KeyError:
            raise HTTPException(404, f"无此关节：{name}")

        cb = CalibrationBody(**{body.field: cur})
        try:
            updated = await asyncio.get_running_loop().run_in_executor(
                None, _yaml_update_joint, config_path, name, cb,
            )
        except KeyError:
            raise HTTPException(404, f"无此关节：{name}")
        await service.reload_config()
        return {"joint": name, "field": body.field, "raw": cur, "fields": updated}

    # ---- 配置：联锁约束 ----

    @app.post("/api/constraints")
    async def api_add_constraint(body: ConstraintBody) -> Dict[str, Any]:
        try:
            added = await asyncio.get_running_loop().run_in_executor(
                None, _yaml_add_constraint, config_path, body,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        await service.reload_config()
        return {"added": added}

    @app.put("/api/constraints/{target}/{gate}")
    async def api_update_constraint(
        target: str, gate: str, body: ConstraintUpdateBody,
    ) -> Dict[str, Any]:
        try:
            updated = await asyncio.get_running_loop().run_in_executor(
                None, _yaml_update_constraint,
                config_path, target, gate, body,
            )
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        await service.reload_config()
        return {"updated": updated}

    @app.delete("/api/constraints/{target}/{gate}")
    async def api_delete_constraint(target: str, gate: str) -> Dict[str, str]:
        ok = await asyncio.get_running_loop().run_in_executor(
            None, _yaml_delete_constraint, config_path, target, gate,
        )
        if not ok:
            raise HTTPException(404, f"未找到约束：{target} ← {gate}")
        await service.reload_config()
        return {"deleted": f"{target}←{gate}"}

    # ---- 配置：前置联动 ----

    @app.post("/api/prerequisites")
    async def api_add_prerequisite(body: PrerequisiteBody) -> Dict[str, Any]:
        try:
            added = await asyncio.get_running_loop().run_in_executor(
                None, _yaml_add_prerequisite, config_path, body,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        await service.reload_config()
        return {"added": added}

    @app.put("/api/prerequisites/{target}/{gate}")
    async def api_update_prerequisite(
        target: str, gate: str, body: PrerequisiteUpdateBody,
    ) -> Dict[str, Any]:
        try:
            updated = await asyncio.get_running_loop().run_in_executor(
                None, _yaml_update_prerequisite,
                config_path, target, gate, body,
            )
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        await service.reload_config()
        return {"updated": updated}

    @app.delete("/api/prerequisites/{target}/{gate}")
    async def api_delete_prerequisite(target: str, gate: str) -> Dict[str, str]:
        ok = await asyncio.get_running_loop().run_in_executor(
            None, _yaml_delete_prerequisite, config_path, target, gate,
        )
        if not ok:
            raise HTTPException(404, f"未找到前置联动：{target} ← {gate}")
        await service.reload_config()
        return {"deleted": f"{target}←{gate}"}

    # ---- 配置：条件限位 ----

    @app.post("/api/zone-limits")
    async def api_add_zone_limit(body: ZoneLimitBody) -> Dict[str, Any]:
        try:
            added = await asyncio.get_running_loop().run_in_executor(
                None, _yaml_add_zone_limit, config_path, body,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        await service.reload_config()
        return {"added": added}

    @app.put("/api/zone-limits/{target}/{gate}")
    async def api_update_zone_limit(
        target: str, gate: str, body: ZoneLimitUpdateBody,
    ) -> Dict[str, Any]:
        try:
            updated = await asyncio.get_running_loop().run_in_executor(
                None, _yaml_update_zone_limit,
                config_path, target, gate, body,
            )
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        await service.reload_config()
        return {"updated": updated}

    @app.delete("/api/zone-limits/{target}/{gate}")
    async def api_delete_zone_limit(target: str, gate: str) -> Dict[str, str]:
        ok = await asyncio.get_running_loop().run_in_executor(
            None, _yaml_delete_zone_limit, config_path, target, gate,
        )
        if not ok:
            raise HTTPException(404, f"未找到条件限位：{target} ← {gate}")
        await service.reload_config()
        return {"deleted": f"{target}←{gate}"}

    @app.post("/api/reload-config")
    async def api_reload() -> Dict[str, str]:
        await service.reload_config()
        return {"status": "reloaded"}

    # ---- WebSocket：实时状态推送 ----

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                if not service.ready:
                    await ws.send_json({"ready": False, "error": service.error})
                else:
                    try:
                        snap = await service.snapshot()
                    except Exception as e:
                        await ws.send_json({"ready": True, "error": str(e)})
                    else:
                        await ws.send_json(snap)
                await asyncio.sleep(0.2)        # 5 Hz
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("WebSocket 异常")

    # ---- 全局错误处理 ----

    @app.exception_handler(RuntimeError)
    async def runtime_err(request: Request, exc: RuntimeError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_err(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="6-DOF Arm Web Console")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--log-level", default="info")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    import uvicorn

    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
