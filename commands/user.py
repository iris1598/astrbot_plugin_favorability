"""
用户指令模块 - UserCommands

提供面向普通用户的好感度相关命令：
  - /查询好感度 [@用户] — 查询好感度（查自己或@他人）
  - /好感度排行 — 查看当前会话的排行榜（前10名，PIL图片）
  - /重置好感度 — 重置自己的好感度
  - /确认关系 — 确认待生效的关系变动提议
  - /取消关系 — 拒绝待生效的关系变动提议
"""

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
import astrbot.api.message_components as Comp

from ..models.manager import extract_user_id


class UserCommands:
    """用户指令集合，通过插件主类绑定。"""

    def __init__(self, plugin_instance):
        self._plugin = plugin_instance

    @property
    def plugin(self):
        return self._plugin

    # ── 查询好感度（统一入口） ─────────────────────────────

    async def cmd_query(self, event: AstrMessageEvent):
        """查询好感度。不带参数查自己，@他人查指定用户。"""
        plug = self.plugin
        group_key, self_id = plug.keys(event)

        # 优先从消息链提取 @ 目标
        target_id = None
        for comp in event.message_obj.message:
            if isinstance(comp, Comp.At):
                target_id = str(comp.qq)
                break

        if target_id:
            label = f"用户 {target_id}"
        else:
            # 无 @ 目标，从文本参数提取
            parts = event.message_str.split()
            if len(parts) >= 2:
                target_id = extract_user_id(parts[1])
                label = f"用户 {target_id}"
            else:
                target_id = self_id
                label = "你"

        info = plug.db.get_user_info(group_key, target_id)
        score = info["score"]
        evaluation = info["eval"]
        relation = info.get("relation") or plug.db.DEFAULT_RELATION
        pending_rel = plug.db.effective_pending(info)

        # 尝试 PIL 图片渲染
        if plug.has_renderer:
            try:
                img_path = plug.renderer.render_favorability_card(
                    user_name=event.get_sender_name()
                    if target_id == self_id
                    else label,
                    user_id=target_id,
                    score=score,
                    evaluation=evaluation,
                    relation=relation,
                )
                yield event.image_result(img_path)
                return
            except Exception as e:
                logger.error(f"[favorability] 查询图片渲染失败，回退文本: {e}")

        msg = (
            f"📊 {label}的好感度档案\n"
            f"分数：{score}\n"
            f"关系：{relation}\n"
            f"评价：{evaluation}"
        )
        if target_id == self_id and pending_rel:
            msg += (
                f"\n⏳ 待确认的关系变动：「{pending_rel['from']}」→"
                f"「{pending_rel['to']}」（回复「确认关系」或「取消关系」）"
            )
        yield event.plain_result(msg)

    # ── 好感度正序（高分在前） ───────────────────────────

    async def cmd_rank(self, event: AstrMessageEvent):
        """查询当前会话的好感度正序排行榜（高分在前，前10名）。"""
        plug = self.plugin
        group_key, _ = plug.keys(event)
        ranked = plug.db.get_ranked_users(group_key, top_n=10, ascending=False)

        if not ranked:
            if plug.has_renderer:
                try:
                    img_path = plug.renderer.render_empty_ranking()
                    yield event.image_result(img_path)
                    return
                except Exception:
                    pass
            yield event.plain_result("🌸 还没有好感度记录哦~")
            return

        if plug.has_renderer:
            try:
                img_path = plug.renderer.render_ranking_image(ranked, ascending=False)
                yield event.image_result(img_path)
                return
            except Exception as e:
                logger.error(f"[favorability] 排行图片渲染失败，回退文本: {e}")

        medals = ["🥇", "🥈", "🥉"] + ["👤"] * 7
        msg = "🏆 【好感度荣誉榜】 🏆\n————————————————"
        for i, (uid, udata) in enumerate(ranked):
            display_eval = (
                (udata["eval"][:12] + "..")
                if len(udata["eval"]) > 12
                else udata["eval"]
            )
            msg += f"\n{medals[i]} {uid} | {udata['score']}分\n   └ 📝 {display_eval}"
        msg += "\n————————————————\n💡 发送「查询好感度」查看你的详细档案"
        yield event.plain_result(msg)

    # ── 好感度倒序（低分在前） ───────────────────────────

    async def cmd_rank_desc(self, event: AstrMessageEvent):
        """查询当前会话的好感度倒序排行榜（低分在前，前10名）。"""
        plug = self.plugin
        group_key, _ = plug.keys(event)
        ranked = plug.db.get_ranked_users(group_key, top_n=10, ascending=True)

        if not ranked:
            if plug.has_renderer:
                try:
                    img_path = plug.renderer.render_empty_ranking()
                    yield event.image_result(img_path)
                    return
                except Exception:
                    pass
            yield event.plain_result("🌸 还没有好感度记录哦~")
            return

        if plug.has_renderer:
            try:
                img_path = plug.renderer.render_ranking_image(ranked, ascending=True)
                yield event.image_result(img_path)
                return
            except Exception as e:
                logger.error(f"[favorability] 倒序排行图片渲染失败，回退文本: {e}")

        medals = ["👤"] * 10
        msg = "📉 【好感度倒序榜】 📉\n————————————————"
        for i, (uid, udata) in enumerate(ranked):
            display_eval = (
                (udata["eval"][:12] + "..")
                if len(udata["eval"]) > 12
                else udata["eval"]
            )
            msg += f"\n{i + 1}. {uid} | {udata['score']}分\n   └ 📝 {display_eval}"
        msg += "\n————————————————\n💡 发送「查询好感度」查看你的详细档案"
        yield event.plain_result(msg)

    # ── 重置自己的好感度 ───────────────────────────────────

    async def cmd_reset_self(self, event: AstrMessageEvent):
        """重置自己的好感度记录。"""
        plug = self.plugin
        group_key, user_id = plug.keys(event)
        await plug.db.reset_user(group_key, user_id)
        yield event.plain_result("✨ 记忆已重置，现在的你对我来说就像一张白纸。")

    # ── 确认/取消关系变动 ──────────────────────────────────

    async def cmd_confirm_relation(self, event: AstrMessageEvent):
        """确认待生效的关系变动提议。"""
        plug = self.plugin
        if not plug.relation_enabled or not plug.favorability_enabled:
            yield event.plain_result("❌ 关系系统未启用。")
            return
        group_key, user_id = plug.keys(event)
        result = await plug.db.confirm_relation(group_key, user_id)
        status = result.get("status")
        if status == "applied":
            up = plug.db.RELATION_LEVELS.index(result["to"]) > plug.db.RELATION_LEVELS.index(
                result["from"]
            )
            emoji = "💞" if up else "💔"
            yield event.plain_result(
                f"{emoji} 关系已变更：「{result['from']}」→「{result['to']}」"
            )
        elif status == "expired":
            yield event.plain_result("⌛ 那个关系提议已经过期失效了。")
        elif status == "stale":
            yield event.plain_result("⚠️ 关系状态已发生变化，该提议已自动作废。")
        else:
            yield event.plain_result("🤔 当前没有待确认的关系变动。")

    async def cmd_cancel_relation(self, event: AstrMessageEvent):
        """拒绝待生效的关系变动提议。"""
        plug = self.plugin
        if not plug.relation_enabled or not plug.favorability_enabled:
            yield event.plain_result("❌ 关系系统未启用。")
            return
        group_key, user_id = plug.keys(event)
        pending = await plug.db.reject_relation(group_key, user_id)
        if pending:
            yield event.plain_result(
                f"🙅 已拒绝对方的关系提议：「{pending['from']}」→「{pending['to']}」，"
                f"关系保持为「{pending['from']}」。"
            )
        else:
            yield event.plain_result("🤔 当前没有待确认的关系变动。")
