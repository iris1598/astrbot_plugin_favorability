"""
好感度数据管理模块 - FavorabilityManager

管理好感度数据的增删改查、持久化、历史数据迁移。
好感度数值（score）与关系（relation）完全解耦：
  - score 只影响 LLM 说话的态度与语气，可随对话自动升降；
  - relation 是独立的 7 档关系，只能通过 LLM 提议 + 用户确认变动，
    且每次只能在相邻档位之间变动。
数据结构：
{
    "group_id_or_private": {
        "user_id": {
            "score": int,
            "relation": str,              # 关系档位名（RELATION_LEVELS 之一）
            "pending_rel": dict | None,   # 待确认的关系变动提议
            "eval": str,
            "name": str,
            "muted_until": float | None   # Unix 时间戳，None 表示未禁言
        }
    }
}
"""

import json
import asyncio
import os
import tempfile
import time
import re
from pathlib import Path
from typing import Optional

from astrbot.api import logger


def extract_user_id(raw: str) -> str:
    """从 @ 提及文本中提取纯数字用户ID。

    支持格式：
      - 纯数字：123456
      - QQ 提及：@昵称(123456)
      - 带 @ 前缀：@123456
    返回提取后的纯数字 ID 字符串。
    """
    raw = raw.strip()
    # 优先从括号中提取数字（QQ @ 提及格式）
    m = re.search(r"\((\d+)\)", raw)
    if m:
        return m.group(1)
    # 去掉前导 @ 后提取数字
    cleaned = raw.lstrip("@")
    m = re.search(r"(\d+)", cleaned)
    if m:
        return m.group(1)
    return raw  # 无法提取时原样返回


def group_storage_key(umo: str, sender_id: str) -> str:
    """把会话隔离(unique_session)的每用户群 UMO 归一化为群级存储键。

    官方隔离开启后群聊 UMO 形如 {平台}:GroupMessage:{用户}_{群}，
    好感度数据（排行/禁言/成员档案）本应按群共享一个桶，因此剥掉
    发话者前缀还原为 {平台}:GroupMessage:{群}。

    - 私聊/webchat 等非群 UMO 原样返回（本就是用户级）。
    - 仅当 session_id 恰好以 "{sender_id}_" 开头时才剥离，
      其他平台（群号本身不含该模式）不受影响。
    """
    parts = umo.split(":", 2)
    if len(parts) != 3 or parts[1] != "GroupMessage":
        return umo
    sid = parts[2]
    prefix = f"{sender_id}_"
    if sender_id and sid.startswith(prefix) and len(sid) > len(prefix):
        return f"{parts[0]}:{parts[1]}:{sid[len(prefix):]}"
    return umo


class FavorabilityManager:
    """好感度数据管理器，负责 CRUD 和持久化。"""

    # 关系档位（从低到高，相邻一档）。与好感度数值解耦。
    RELATION_LEVELS = (
        "决裂陌路",
        "不合对头",
        "生疏之交",
        "普通朋友",
        "熟络好友",
        "知心挚友",
        "挚爱恋人",
    )
    DEFAULT_RELATION = "普通朋友"
    RELATION_PENDING_TTL = 600  # 关系变动提议的有效期（秒）
    RELATION_COOLDOWN = 600  # 关系被取消/拒绝后的冷却时间（秒）

    # 历史旧档位映射表（平滑迁移与兼容输入）
    LEGACY_RELATION_MAP = {
        "亲密无间": "挚爱恋人",
        "亲密朋友": "知心挚友",
        "聊得来的熟人": "熟络好友",
        "普通关系": "普通朋友",
        "心存芥蒂": "生疏之交",
        "明显反感": "不合对头",
        "关系破裂": "决裂陌路",
    }

    DEFAULT_USER = {
        "score": 0,
        "relation": DEFAULT_RELATION,
        "pending_rel": None,
        "eval": "初次见面",
        "name": "",
        "muted_until": None,
        "rel_cooldown_until": None,
    }
    MUTE_MAX_SECONDS = 300  # 最长禁言 5 分钟

    @classmethod
    def relation_for_score(cls, score: int) -> str:
        """仅供旧数据迁移使用：按历史分数区间推导初始关系。"""
        if score >= 70:
            return "挚爱恋人"
        if score >= 50:
            return "知心挚友"
        if score >= 21:
            return "熟络好友"
        if score >= -20:
            return "普通朋友"
        if score >= -50:
            return "生疏之交"
        if score >= -70:
            return "不合对头"
        return "决裂陌路"

    @classmethod
    def next_relation(cls, current: str, direction: str) -> Optional[str]:
        """返回相邻一档的目标关系名；越界或档位名非法返回 None。"""
        current = cls.LEGACY_RELATION_MAP.get(current, current)
        try:
            index = cls.RELATION_LEVELS.index(current)
        except ValueError:
            index = cls.RELATION_LEVELS.index(cls.DEFAULT_RELATION)
        if direction == "up":
            target = index + 1
        elif direction == "down":
            target = index - 1
        else:
            return None
        if 0 <= target < len(cls.RELATION_LEVELS):
            return cls.RELATION_LEVELS[target]
        return None

    @staticmethod
    def effective_pending(user_data: dict) -> Optional[dict]:
        """返回未过期的待确认关系提议，已过期或没有返回 None。"""
        pending = user_data.get("pending_rel")
        if not isinstance(pending, dict):
            return None
        if time.time() > pending.get("expires_at", 0):
            return None
        return pending

    def __init__(self, data_path: Path):
        self.data_file = data_path / "favorability.json"
        self.lock = asyncio.Lock()
        data_path.mkdir(parents=True, exist_ok=True)
        if not self.data_file.exists():
            self._write({})
        # 启动时自动迁移：修正历史错误格式的 user_id key
        self._migrate_legacy_keys()

    def _read(self) -> dict:
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write(self, data: dict):
        tmp_path = None
        try:
            # 写入同目录下的临时文件，然后原子替换，防止意外关机/断电导致损坏
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self.data_file.parent, prefix="fav_tmp_", suffix=".json"
            )
            with open(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.data_file)
        except Exception as e:
            logger.error(f"[favorability] 写入失败: {e}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _migrate_legacy_keys(self):
        """启动时迁移历史错误 key（如 @昵称(123456)）为纯数字 ID，并补充缺失的 name 字段。"""
        data = self._read()
        migrated = 0
        patched = 0
        new_data = {}
        for group_key, users in data.items():
            if not isinstance(users, dict):
                new_data[group_key] = users
                continue
            new_users = {}
            for old_key, val in users.items():
                new_key = extract_user_id(old_key)
                # 填充缺失的 name 字段
                if isinstance(val, dict) and "name" not in val:
                    val["name"] = ""
                    patched += 1
                # 填充缺失的 muted_until 字段
                if isinstance(val, dict) and "muted_until" not in val:
                    val["muted_until"] = None
                    patched += 1
                # 解耦升级：为旧数据补充 relation（按历史分数区间推导）或平滑迁移旧档名
                if isinstance(val, dict):
                    rel = val.get("relation")
                    if not rel:
                        val["relation"] = self.relation_for_score(
                            int(val.get("score", 0) or 0)
                        )
                        patched += 1
                    elif rel in self.LEGACY_RELATION_MAP:
                        val["relation"] = self.LEGACY_RELATION_MAP[rel]
                        patched += 1
                if isinstance(val, dict) and "pending_rel" not in val:
                    val["pending_rel"] = None
                    patched += 1
                if isinstance(val, dict) and "rel_cooldown_until" not in val:
                    val["rel_cooldown_until"] = None
                    patched += 1
                # 如果同一 group 内新 key 已存在，保留 score 较大的
                if new_key in new_users:
                    if val.get("score", 0) > new_users[new_key].get("score", 0):
                        new_users[new_key] = val
                else:
                    new_users[new_key] = val
                if new_key != old_key:
                    migrated += 1
            new_data[group_key] = new_users
        if migrated > 0 or patched > 0:
            self._write(new_data)
            details = []
            if migrated:
                details.append(f"修正了 {migrated} 条历史错误 key")
            if patched:
                details.append(f"补填/升级了 {patched} 条缺失字段")
            logger.info(f"[favorability] 数据迁移完成：{'；'.join(details)}")

    # ── 业务方法 ────────────────────────────────────────────

    def parse_origin(self, group_key: str, user_id: str) -> tuple[str, str]:
        """返回 (group_key, user_id) — 保留以供外部构造使用。"""
        return group_key, user_id

    def get_user_info(self, group_key: str, user_id: str) -> dict:
        data = self._read()
        raw = data.get(group_key, {}).get(user_id, self.DEFAULT_USER.copy())
        # 确保所有字段存在（兼容旧数据）
        result = self.DEFAULT_USER.copy()
        result.update(raw)
        if result.get("relation") in self.LEGACY_RELATION_MAP:
            result["relation"] = self.LEGACY_RELATION_MAP[result["relation"]]
        return result

    async def update_user(
        self,
        group_key: str,
        user_id: str,
        change: int = 0,
        new_eval: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> dict:
        async with self.lock:
            data = self._read()
            if group_key not in data:
                data[group_key] = {}
            if user_id not in data[group_key]:
                data[group_key][user_id] = self.DEFAULT_USER.copy()
            data[group_key][user_id]["score"] += max(-5, min(5, change))
            if new_eval:
                data[group_key][user_id]["eval"] = new_eval.strip()
            if user_name:
                data[group_key][user_id]["name"] = user_name
            self._write(data)
            return data[group_key][user_id]

    async def set_score(
        self, group_key: str, user_id: str, score: int, user_name: Optional[str] = None
    ):
        async with self.lock:
            data = self._read()
            if group_key not in data:
                data[group_key] = {}
            if user_id not in data[group_key]:
                data[group_key][user_id] = self.DEFAULT_USER.copy()
            data[group_key][user_id]["score"] = score
            if user_name:
                data[group_key][user_id]["name"] = user_name
            self._write(data)

    # ── 关系变动（提议 → 用户确认） ─────────────────────────

    async def set_relation(
        self,
        group_key: str,
        user_id: str,
        relation: str,
        user_name: Optional[str] = None,
    ) -> bool:
        """直接设置关系档位（管理员指令用）。档位名非法返回 False。"""
        relation = self.LEGACY_RELATION_MAP.get(relation, relation)
        if relation not in self.RELATION_LEVELS:
            return False
        async with self.lock:
            data = self._read()
            if group_key not in data:
                data[group_key] = {}
            if user_id not in data[group_key]:
                data[group_key][user_id] = self.DEFAULT_USER.copy()
            data[group_key][user_id]["relation"] = relation
            data[group_key][user_id]["pending_rel"] = None
            if user_name:
                data[group_key][user_id]["name"] = user_name
            self._write(data)
        return True

    async def propose_relation(
        self,
        group_key: str,
        user_id: str,
        direction: str,
        user_name: Optional[str] = None,
    ) -> dict:
        """记录一次相邻一档的关系变动提议，等待用户确认。

        Returns:
            {"status": "ok"|"already_pending"|"cooldown"|"low_score_rejected"|"boundary", "pending": ..., "current": ...}
        """
        async with self.lock:
            data = self._read()
            if group_key not in data:
                data[group_key] = {}
            if user_id not in data[group_key]:
                data[group_key][user_id] = self.DEFAULT_USER.copy()
            user = data[group_key][user_id]
            current = user.get("relation") or self.DEFAULT_RELATION
            pending = self.effective_pending(user)
            if pending is not None:
                return {"status": "already_pending", "pending": pending}

            # 冷却检查：被取消或拒绝后在冷却期内不接受重复提议
            cooldown_until = user.get("rel_cooldown_until")
            if cooldown_until and time.time() < cooldown_until:
                return {
                    "status": "cooldown",
                    "remaining": max(1, int(cooldown_until - time.time())),
                    "current": current,
                }

            # 基础合理性防幻觉：好感度为负数时不允许发起升档提议
            current_score = int(user.get("score", 0) or 0)
            if direction == "up" and current_score < 0:
                return {
                    "status": "low_score_rejected",
                    "current_score": current_score,
                    "current": current,
                }

            target = self.next_relation(current, direction)
            if target is None:
                return {"status": "boundary", "current": current}
            new_pending = {
                "direction": direction,
                "from": current,
                "to": target,
                "expires_at": time.time() + self.RELATION_PENDING_TTL,
            }
            user["pending_rel"] = new_pending
            if user_name:
                user["name"] = user_name
            self._write(data)
            return {"status": "ok", "pending": new_pending}

    async def confirm_relation(self, group_key: str, user_id: str) -> dict:
        """用户确认待生效的关系变动。

        Returns:
            {"status": "applied", "from": 旧档, "to": 新档} 或
            {"status": "none"|"expired"|"stale"}
        """
        async with self.lock:
            data = self._read()
            user = data.get(group_key, {}).get(user_id)
            if not isinstance(user, dict) or not user.get("pending_rel"):
                return {"status": "none"}
            pending = user["pending_rel"]
            user["pending_rel"] = None
            if time.time() > pending.get("expires_at", 0):
                self._write(data)
                return {"status": "expired"}
            current = user.get("relation") or self.DEFAULT_RELATION
            if pending.get("from") != current or pending.get("to") not in self.RELATION_LEVELS:
                # 提议后关系被其他方式改动，本次提议作废
                self._write(data)
                return {"status": "stale"}
            user["relation"] = pending["to"]
            user["rel_cooldown_until"] = None
            self._write(data)
            return {"status": "applied", "from": current, "to": pending["to"]}

    async def reject_relation(self, group_key: str, user_id: str) -> Optional[dict]:
        """用户拒绝（取消）待确认的关系变动提议。返回被取消的提议或 None。"""
        async with self.lock:
            data = self._read()
            user = data.get(group_key, {}).get(user_id)
            if not isinstance(user, dict) or not user.get("pending_rel"):
                return None
            pending = user["pending_rel"]
            user["pending_rel"] = None
            # 拒绝后进入冷静期，避免连续被同方向提议骚扰
            user["rel_cooldown_until"] = time.time() + self.RELATION_COOLDOWN
            self._write(data)
            return self.effective_pending({"pending_rel": pending})

    async def reset_user(
        self, group_key: str, user_id: str, user_name: Optional[str] = None
    ):
        async with self.lock:
            data = self._read()
            if group_key in data and user_id in data[group_key]:
                data[group_key][user_id] = {
                    "score": 0,
                    "relation": self.DEFAULT_RELATION,
                    "pending_rel": None,
                    "eval": "记忆已被抹除",
                    "name": user_name or data[group_key][user_id].get("name", ""),
                    "muted_until": None,
                    "rel_cooldown_until": None,
                }
                self._write(data)

    def get_group_data(self, group_key: str) -> dict:
        return self._read().get(group_key, {})

    def get_ranked_users(
        self, group_key: str, top_n: int = 10, ascending: bool = False
    ) -> list[tuple[str, dict]]:
        """获取当前群组好感度排行（前 top_n 名）。

        Args:
            ascending: False=倒序（高分在前）, True=正序（低分在前）
        """
        group_data = self.get_group_data(group_key)
        safe = {}
        for uid, udata in group_data.items():
            entry = self.DEFAULT_USER.copy()
            entry.update(udata)
            safe[uid] = entry
        sorted_list = sorted(
            safe.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=not ascending,
        )
        return sorted_list[:top_n]

    # ── 禁言相关方法 ────────────────────────────────────────

    def is_muted(self, group_key: str, user_id: str) -> bool:
        """检查用户是否处于禁言状态（自动清除已过期）。"""
        data = self._read()
        user_data = data.get(group_key, {}).get(user_id)
        if not user_data:
            return False
        muted_until = user_data.get("muted_until")
        if muted_until is None:
            return False
        if time.time() > muted_until:
            return False  # 已过期
        return True

    def get_mute_remaining(self, group_key: str, user_id: str) -> float:
        """获取剩余禁言秒数，未禁言返回 0。"""
        data = self._read()
        user_data = data.get(group_key, {}).get(user_id)
        if not user_data:
            return 0
        muted_until = user_data.get("muted_until")
        if muted_until is None:
            return 0
        remaining = muted_until - time.time()
        return max(0, remaining)

    async def mute_user(
        self, group_key: str, user_id: str, seconds: int
    ) -> float:
        """禁言用户指定秒数（最长 5 分钟）。

        Returns:
            muted_until 时间戳
        """
        seconds = min(max(1, seconds), self.MUTE_MAX_SECONDS)
        muted_until = time.time() + seconds
        async with self.lock:
            data = self._read()
            if group_key not in data:
                data[group_key] = {}
            if user_id not in data[group_key]:
                data[group_key][user_id] = self.DEFAULT_USER.copy()
            data[group_key][user_id]["muted_until"] = muted_until
            self._write(data)
        return muted_until

    async def unmute_user(self, group_key: str, user_id: str):
        """解除用户禁言。"""
        async with self.lock:
            data = self._read()
            if group_key in data and user_id in data[group_key]:
                data[group_key][user_id]["muted_until"] = None
                self._write(data)

