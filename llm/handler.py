"""
LLM 请求/响应处理模块 - LLMHandler

职责：
1. on_llm_request：向 LLM 注入好感度系统规则（静态→system_prompt）与动态状态（→extra_user_content_parts）
   同时检查禁言状态，若被禁言则阻断 LLM 请求并发送"不理你"式回复
2. on_llm_response：解析 LLM 响应中的 FAV/EVAL/STK/MUTE 标签，更新数据库并异步发送补充消息
"""

import asyncio
import random
from datetime import datetime

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core.agent.message import TextPart

from ..services.prompt import (
    PromptManager,
    RE_FAV,
    RE_EVAL,
    RE_STK,
    RE_MUTE,
    clean_tags_from_text,
    validate_fav_value,
    validate_eval_text,
    validate_mute_seconds,
)

# 禁言时的回复模板（随机选择一条）
MUTE_REPLIES = [
    "哼！不想理你了！(生气地转过身去)",
    "（假装听不见）啦啦啦~",
    "我听不见我听不见~略略略！",
    "生气了！哄不好的那种！",
    "不要和你说话了！(╯‵□′)╯︵┻━┻",
    "你走开！我不想看到你！",
    "（捂住耳朵）不听不听，王八念经！",
    "哼唧…等你变乖了再来找我吧！",
    "我现在很生气，后果很严重！不想理你！",
    "（背过身去）不要跟我讲话！",
]


class LLMHandler:
    """封装 LLM 请求注入和响应解析逻辑。"""

    def __init__(self, plugin_instance):
        # 以弱引用方式持有插件实例，避免循环引用
        self._plugin = plugin_instance

    @property
    def plugin(self):
        return self._plugin

    # ── on_llm_request ─────────────────────────────────────

    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """向 LLM 注入好感度系统规则与动态状态；若用户被禁言则阻断请求。"""
        plug = self.plugin

        # ── 禁言检查（优先级最高） ─────────────────────────
        group_key, user_id = plug.keys(event)
        is_muted = (
            plug.db.is_muted(group_key, user_id)
            if plug.mute_enabled
            else False
        )
        mute_remaining = (
            plug.db.get_mute_remaining(group_key, user_id)
            if is_muted
            else 0
        )

        if is_muted and mute_remaining > 0:
            # 阻断 LLM 请求，发送"不理你"回复
            logger.info(
                f"[favorability] 用户 {user_id} 处于禁言状态（剩余 {int(mute_remaining)}s），阻断 LLM 请求"
            )
            event.stop_event()
            # 随机选择一条禁言回复
            reply = random.choice(MUTE_REPLIES)
            try:
                await event.send(event.plain_result(reply))
            except Exception as e:
                logger.error(f"[favorability] 禁言回复发送失败: {e}")
            return

        if not plug.favorability_enabled and not plug.sticker_enabled:
            return

        # 第一部分：静态规则 → system_prompt
        static_prompt = PromptManager.build_static_prompt(
            favorability_enabled=plug.favorability_enabled,
            sticker_enabled=plug.sticker_enabled,
            sticker_categories=(
                plug.stickers.get_categories() if plug.sticker_enabled else None
            ),
            mute_condition=plug.mute_condition,
        )
        if static_prompt:
            req.system_prompt = (req.system_prompt or "") + static_prompt

        # 第二部分：动态状态 → extra_user_content_parts
        user_info = None
        if plug.favorability_enabled:
            user_info = plug.db.get_user_info(group_key, user_id)

        dynamic_text = PromptManager.build_dynamic_context(
            favorability_enabled=plug.favorability_enabled,
            system_time_enabled=plug.system_time_enabled,
            user_info_enabled=plug.user_info_enabled,
            score=user_info.get("score") if user_info else None,
            eval_text=user_info.get("eval") if user_info else None,
            time_str=(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if plug.system_time_enabled
                else None
            ),
            sender_name=event.get_sender_name() if plug.user_info_enabled else None,
            sender_id=event.get_sender_id() if plug.user_info_enabled else None,
            is_muted=is_muted,
            mute_remaining=mute_remaining,
        )
        if dynamic_text:
            req.extra_user_content_parts.append(TextPart(text=dynamic_text))

    # ── on_llm_response ────────────────────────────────────

    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """解析 LLM 响应中的好感度标记与禁言标记，更新数据库并发送补充消息。"""
        if not resp.completion_text:
            return

        original = resp.completion_text

        # 1. 提取标记
        fav_match = RE_FAV.search(original)
        eval_match = RE_EVAL.search(original)
        stk_matches = RE_STK.findall(original)
        mute_match = RE_MUTE.search(original)

        # 2. 清理文本（移除所有标记）
        clean_text = clean_tags_from_text(original)
        resp.completion_text = clean_text

        if not fav_match and not eval_match and not stk_matches and not mute_match:
            return

        plug = self.plugin
        group_key, user_id = plug.keys(event)

        handle_favor = plug.favorability_enabled and (fav_match or eval_match)
        handle_sticker = plug.sticker_enabled and bool(stk_matches)
        handle_mute = plug.mute_enabled and bool(mute_match)

        if not handle_favor and not handle_sticker and not handle_mute:
            return

        # 3. 处理禁言
        if handle_mute:
            raw_seconds = int(mute_match.group(1))
            if validate_mute_seconds(raw_seconds):
                muted_until = await plug.db.mute_user(
                    group_key, user_id, raw_seconds
                )
                logger.info(
                    f"[favorability] 用户 {user_id} 被禁言 {raw_seconds}s "
                    f"(直到 {muted_until})"
                )
                # 异步发送禁言通知
                asyncio.create_task(
                    self._send_mute_notice(event, plug, raw_seconds)
                )
            else:
                logger.warning(
                    f"[favorability] 过滤非法 MUTE 值: {raw_seconds}s"
                )

        # 4. 解析并验证 FAV 值
        raw_change = int(fav_match.group(1)) if fav_match else 0
        if raw_change != 0 and not validate_fav_value(raw_change):
            # 超出范围则忽略 FAV 标记，仅保留 EVAL
            raw_change = 0
            logger.warning(f"[favorability] 过滤非法 FAV 值: {raw_change}，仅处理 EVAL")
        change = max(-5, min(5, raw_change))

        # 5. 解析并验证 EVAL
        new_eval = None
        if eval_match:
            raw_eval = eval_match.group(1).strip()
            if validate_eval_text(raw_eval):
                new_eval = raw_eval
            else:
                logger.warning(f"[favorability] 过滤非法 EVAL 文本: {raw_eval[:30]}")

        # 6. 校验：如果 change == 0 且 new_eval 为 None，说明无有效操作
        if change == 0 and new_eval is None and not stk_matches:
            return

        # 7. 更新数据库
        if handle_favor:
            if change != 0 or new_eval is not None:
                user_data = await plug.db.update_user(
                    group_key,
                    user_id,
                    change,
                    new_eval,
                    user_name=event.get_sender_name(),
                )
            else:
                user_data = plug.db.get_user_info(group_key, user_id)
        else:
            user_data = None

        # 8. 异步补发提示与表情包
        asyncio.create_task(
            self._send_extra_messages(
                event,
                plug,
                handle_favor,
                handle_sticker,
                change,
                new_eval,
                user_data,
                stk_matches,
            )
        )

    # ── 异步辅助：发送补充消息 ─────────────────────────────

    async def _send_extra_messages(
        self,
        event: AstrMessageEvent,
        plug,
        handle_favor: bool,
        handle_sticker: bool,
        change: int,
        new_eval: str | None,
        user_data: dict | None,
        stk_matches: list[str],
    ):
        """延迟发送好感度变化提示和表情包。"""
        await asyncio.sleep(0.5)
        umo = event.unified_msg_origin

        if handle_favor and user_data:
            tips = []
            if change != 0:
                symbol = "+" if change > 0 else ""
                tips.append(f"好感度 {symbol}{change}（当前: {user_data['score']}）")
            if new_eval is not None:
                tips.append("评价已更新 ✨")
            if tips:
                try:
                    mc = MessageChain().message(" | ".join(tips))
                    await plug.context.send_message(umo, mc)
                except Exception as e:
                    logger.error(f"[favorability] 提示发送失败: {e}")

        if handle_sticker:
            for cat in stk_matches:
                img_path = plug.stickers.get_random_sticker(cat.strip())
                if img_path:
                    try:
                        mc = MessageChain().file_image(str(img_path))
                        await plug.context.send_message(umo, mc)
                    except Exception as e:
                        logger.error(f"[favorability] 表情包发送失败: {e}")

    async def _send_mute_notice(
        self,
        event: AstrMessageEvent,
        plug,
        seconds: int,
    ):
        """异步发送禁言通知。"""
        await asyncio.sleep(0.5)
        umo = event.unified_msg_origin
        try:
            mc = MessageChain().message(
                f"🔇 你已被禁言 {seconds} 秒，在此期间不能对话。"
            )
            await plug.context.send_message(umo, mc)
        except Exception as e:
            logger.error(f"[favorability] 禁言通知发送失败: {e}")
