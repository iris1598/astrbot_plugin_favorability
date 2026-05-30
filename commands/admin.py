"""
管理员指令模块 - AdminCommands

提供面向管理员的指令：
  - /设置好感度 <@用户> <分数> — 强制设置指定用户好感度
  - /查询好感度 [@用户] — 查询指定用户好感度（渲染为图片）
  - /重置指定好感度 <@用户> — 重置指定用户好感度
"""

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
import astrbot.api.message_components as Comp

from ..models.manager import extract_user_id


class AdminCommands:
    """管理员指令集合，通过插件主类绑定。"""

    def __init__(self, plugin_instance):
        self._plugin = plugin_instance

    @property
    def plugin(self):
        return self._plugin

    def _extract_at_user(self, event: AstrMessageEvent) -> str | None:
        """从消息链中提取 @ 目标的纯数字 ID。"""
        for comp in event.message_obj.message:
            if isinstance(comp, Comp.At):
                return str(comp.qq)
        return None

    # ── 设置好感度 ─────────────────────────────────────────

    async def cmd_admin_set(self, event: AstrMessageEvent):
        """(管理员) 强制设置指定用户好感度。用法: /设置好感度 <@用户> <分数>"""
        if event.role != "admin":
            yield event.plain_result("❌ 此命令仅限管理员使用。")
            return

        target_id = self._extract_at_user(event)
        text = event.message_str
        parts = text.split()

        if target_id:
            if len(parts) < 2:
                yield event.plain_result("❌ 用法: /设置好感度 <@用户> <分数>")
                return
            try:
                score_val = int(parts[-1])
            except ValueError:
                yield event.plain_result("❌ 分数必须是整数。")
                return
        else:
            # 兼容旧格式：纯文本方式 /设置好感度 用户ID 分数
            if len(parts) < 3:
                yield event.plain_result("❌ 用法: /设置好感度 <@用户> <分数>")
                return
            target_id = extract_user_id(parts[1])
            try:
                score_val = int(parts[2])
            except ValueError:
                yield event.plain_result("❌ 分数必须是整数。")
                return

        if not target_id or not target_id.isdigit():
            yield event.plain_result("❌ 无法识别用户 ID。")
            return

        group_key, _ = self.plugin.keys(event)
        sender_name = event.get_sender_name()
        await self.plugin.db.set_score(
            group_key, target_id, score_val, user_name=sender_name
        )
        yield event.plain_result(f"✅ 已将用户 {target_id} 的好感度设为 {score_val}。")

    # ── 重置指定用户好感度 ─────────────────────────────────

    async def cmd_admin_reset(self, event: AstrMessageEvent):
        """(管理员) 重置指定用户的好感度。用法: /重置指定好感度 <@用户>"""
        if event.role != "admin":
            yield event.plain_result("❌ 此命令仅限管理员使用。")
            return

        target_id = self._extract_at_user(event)

        if target_id is None:
            parts = event.message_str.split()
            if len(parts) < 2:
                yield event.plain_result("❌ 用法: /重置指定好感度 <@用户>")
                return
            target_id = extract_user_id(parts[1])

        if not target_id or not target_id.isdigit():
            yield event.plain_result("❌ 无法识别用户 ID。")
            return

        group_key, _ = self.plugin.keys(event)
        await self.plugin.db.reset_user(group_key, target_id)
        yield event.plain_result(f"✅ 用户 {target_id} 的好感度已重置。")
