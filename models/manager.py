"""
好感度数据管理模块 - FavorabilityManager

管理好感度数据的增删改查、持久化、历史数据迁移。
数据结构：
{
    "group_id_or_private": {
        "user_id": {"score": int, "eval": str}
    }
}
"""

import json
import asyncio
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


class FavorabilityManager:
    """好感度数据管理器，负责 CRUD 和持久化。"""

    DEFAULT_USER = {"score": 0, "eval": "初次见面", "name": ""}

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
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[favorability] 写入失败: {e}")

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
                details.append(f"补填了 {patched} 条缺失字段")
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

    async def reset_user(
        self, group_key: str, user_id: str, user_name: Optional[str] = None
    ):
        async with self.lock:
            data = self._read()
            if group_key in data and user_id in data[group_key]:
                data[group_key][user_id] = {
                    "score": 0,
                    "eval": "记忆已被抹除",
                    "name": user_name or data[group_key][user_id].get("name", ""),
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
