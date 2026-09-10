"""
LLM 请求/响应处理模块 - LLMHandler

职责：
1. on_llm_request：向 LLM 注入好感度系统规则（静态→system_prompt，含当前关系档位的
   行为准则）与动态状态（→extra_user_content_parts）；
   同时检查禁言状态，若被禁言则阻断 LLM 请求并发送"不理你"式回复
2. on_llm_response：解析 LLM 响应中的 FAV/EVAL/REL/STK/MUTE 标签，更新数据库并异步发送补充消息；
   REL 标签仅创建待确认的关系变动提议，需用户使用「确认关系」指令确认后才生效
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
    RE_REL,
    clean_tags_from_text,
    normalize_rel_direction,
    validate_fav_value,
    validate_eval_text,
    validate_mute_seconds,
    format_system_time,
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
        persona_id = await plug.resolve_persona_id(event, req)
        pconf = plug.get_persona_config(persona_id)

        # ── 插件总开关 ──
        if not pconf.plugin_enabled:
            return

        # ── 禁言检查（优先级最高，按人格隔离） ─────────────────
        group_key, user_id = plug.keys(event)
        is_muted = (
            plug.db.is_muted(group_key, user_id, persona_id=persona_id)
            if pconf.mute_enabled
            else False
        )
        mute_remaining = (
            plug.db.get_mute_remaining(group_key, user_id, persona_id=persona_id)
            if is_muted
            else 0
        )

        if is_muted and mute_remaining > 0:
            # 阻断 LLM 请求，发送"不理你"回复
            logger.info(
                f"[favorability][{persona_id}] 用户 {user_id} 处于禁言状态（剩余 {int(mute_remaining)}s），阻断 LLM 请求"
            )
            event.stop_event()
            # 随机选择一条禁言回复
            reply = random.choice(MUTE_REPLIES)
            try:
                await event.send(event.plain_result(reply))
            except Exception as e:
                logger.error(f"[favorability] 禁言回复发送失败: {e}")
            return

        has_any_feature = (
            pconf.favorability_enabled
            or pconf.sticker_enabled
            or pconf.mute_enabled
            or plug.system_time_enabled
            or plug.user_info_enabled
        )
        if not has_any_feature:
            return

        # 先读取用户在该人格下的状态（关系档位与待确认提议需要按用户注入）
        user_info = None
        if pconf.favorability_enabled:
            user_info = plug.db.get_user_info(
                group_key, user_id, persona_id=persona_id
            )

        # old 预设保持旧版“分数即关系”一体化规则，不启用关系系统
        relation_active = (
            pconf.favorability_enabled
            and pconf.relation_enabled
            and pconf.prompt_preset != "old"
        )

        # 第一部分：静态规则 → system_prompt（只注入当前人格、当前关系档位的准则）
        static_prompt = PromptManager.build_static_prompt(
            favorability_enabled=pconf.favorability_enabled,
            sticker_enabled=pconf.sticker_enabled,
            sticker_categories=(
                plug.stickers.get_categories(
                    persona_id=persona_id, dir_name=pconf.sticker_dir_name
                )
                if pconf.sticker_enabled
                else None
            ),
            mute_condition=pconf.mute_condition,
            mute_enabled=pconf.mute_enabled,
            favorability_prompt_core=pconf.favorability_prompt_core,
            favorability_prompt_behavior=pconf.favorability_prompt_behavior,
            favorability_prompt_relation=pconf.favorability_prompt_relation,
            favorability_prompt_mute=pconf.favorability_prompt_mute,
            favorability_prompt_security=pconf.favorability_prompt_security,
            sticker_condition=pconf.sticker_condition,
            prompt_preset=pconf.prompt_preset,
            relation_enabled=relation_active,
            relation=(user_info or {}).get("relation", ""),
            relation_guidelines=pconf.relation_guidelines,
            plugin_enabled=pconf.plugin_enabled,
        )
        if static_prompt:
            req.system_prompt = (req.system_prompt or "") + static_prompt

        # 第二部分：动态状态 → extra_user_content_parts
        dynamic_text = PromptManager.build_dynamic_context(
            favorability_enabled=pconf.favorability_enabled,
            system_time_enabled=plug.system_time_enabled,
            user_info_enabled=plug.user_info_enabled,
            score=user_info.get("score") if user_info else None,
            eval_text=user_info.get("eval") if user_info else None,
            relation=user_info.get("relation") if user_info else None,
            relation_enabled=relation_active,
            pending_rel=(
                plug.db.effective_pending(user_info) if user_info else None
            ),
            time_str=(
                format_system_time(datetime.now())
                if plug.system_time_enabled
                else None
            ),
            sender_name=event.get_sender_name() if plug.user_info_enabled else None,
            sender_id=event.get_sender_id() if plug.user_info_enabled else None,
            is_muted=is_muted,
            mute_remaining=mute_remaining,
            interaction_hint_enabled=plug.interaction_hint_enabled,
            interaction_hint_text=plug.interaction_hint_text,
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
        rel_match = RE_REL.search(original)

        # 2. 清理文本（移除所有标记）
        clean_text = clean_tags_from_text(original)
        resp.completion_text = clean_text

        if (
            not fav_match
            and not eval_match
            and not stk_matches
            and not mute_match
            and not rel_match
        ):
            return

        plug = self.plugin
        persona_id = await plug.resolve_persona_id(event)
        pconf = plug.get_persona_config(persona_id)
        group_key, user_id = plug.keys(event)

        if not pconf.plugin_enabled:
            return

        handle_favor = pconf.favorability_enabled and (fav_match or eval_match)
        handle_sticker = pconf.sticker_enabled and bool(stk_matches)
        handle_mute = pconf.mute_enabled and bool(mute_match)
        handle_rel = (
            pconf.favorability_enabled
            and pconf.relation_enabled
            and pconf.prompt_preset != "old"
            and bool(rel_match)
        )

        if not handle_favor and not handle_sticker and not handle_mute and not handle_rel:
            return

        # 2.5 处理关系变动提议：只创建“待用户确认”的提议，不直接改关系
        rel_result = None
        if handle_rel:
            direction = normalize_rel_direction(rel_match.group(1))
            if direction:
                rel_result = await plug.db.propose_relation(
                    group_key,
                    user_id,
                    direction,
                    user_name=event.get_sender_name(),
                    persona_id=persona_id,
                )
                if rel_result.get("status") == "ok":
                    p = rel_result["pending"]
                    logger.info(
                        f"[favorability][{persona_id}] 用户 {user_id} 关系提议待确认: "
                        f"{p['from']} -> {p['to']} ({direction})"
                    )
                elif rel_result.get("status") == "low_score_rejected":
                    logger.info(
                        f"[favorability][{persona_id}] 用户 {user_id} 好感度为负({rel_result.get('current_score')})，拦截异常升级提议"
                    )
                elif rel_result.get("status") == "cooldown":
                    logger.debug(
                        f"[favorability][{persona_id}] 用户 {user_id} 关系提议处于冷却中(剩余 {rel_result.get('remaining')}s)"
                    )
            else:
                logger.warning(
                    f"[favorability][{persona_id}] 无法识别的 REL 方向: {rel_match.group(1)}"
                )

        # 3. 处理禁言
        if handle_mute:
            raw_seconds = int(mute_match.group(1))
            if validate_mute_seconds(raw_seconds):
                muted_until = await plug.db.mute_user(
                    group_key, user_id, raw_seconds, persona_id=persona_id
                )
                logger.info(
                    f"[favorability][{persona_id}] 用户 {user_id} 被禁言 {raw_seconds}s "
                    f"(直到 {muted_until})"
                )
                # 异步发送禁言通知
                asyncio.create_task(
                    self._send_mute_notice(event, plug, raw_seconds)
                )
            else:
                logger.warning(
                    f"[favorability][{persona_id}] 过滤非法 MUTE 值: {raw_seconds}s"
                )

        # 4. 解析并验证 FAV 值。兼容并清理模型误输出的“±0”，但不执行更新。
        raw_fav = fav_match.group(1) if fav_match else None
        raw_change = 0 if not raw_fav or raw_fav.startswith("±") else int(raw_fav)
        if raw_fav and (raw_fav.startswith("±") or not validate_fav_value(raw_change)):
            # 超出范围则忽略 FAV 标记，仅保留 EVAL
            raw_change = 0
            logger.warning(f"[favorability][{persona_id}] 过滤非法 FAV 值: {raw_fav}，仅处理 EVAL")
        change = max(-5, min(5, raw_change))

        # 5. 解析并验证 EVAL
        new_eval = None
        if eval_match:
            raw_eval = eval_match.group(1).strip()
            if validate_eval_text(raw_eval):
                new_eval = raw_eval
            else:
                logger.warning(f"[favorability][{persona_id}] 过滤非法 EVAL 文本: {raw_eval[:30]}")

        # 6. 校验：如果 change == 0 且 new_eval 为 None，说明无有效操作
        if change == 0 and new_eval is None and not stk_matches and rel_result is None:
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
                    persona_id=persona_id,
                )
            else:
                user_data = plug.db.get_user_info(
                    group_key, user_id, persona_id=persona_id
                )
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
                rel_result,
                persona_id=persona_id,
                sticker_dir_name=pconf.sticker_dir_name,
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
        rel_result: dict | None = None,
        persona_id: str = "default",
        sticker_dir_name: str | None = None,
    ):
        """延迟发送好感度变化提示、关系变动确认请求和表情包。"""
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

        if rel_result:
            text = self._build_rel_notice(rel_result)
            if text:
                try:
                    mc = MessageChain().message(text)
                    await plug.context.send_message(umo, mc)
                except Exception as e:
                    logger.error(f"[favorability] 关系确认提示发送失败: {e}")

        if handle_sticker:
            for cat in stk_matches:
                img_path = plug.stickers.get_random_sticker(
                    cat.strip(), persona_id=persona_id, dir_name=sticker_dir_name
                )
                if img_path:
                    try:
                        mc = MessageChain().file_image(str(img_path))
                        await plug.context.send_message(umo, mc)
                    except Exception as e:
                        logger.error(f"[favorability] 表情包发送失败: {e}")

    def _build_rel_notice(self, rel_result: dict) -> str | None:
        """根据关系提议结果生成发给用户的确认提示文本。"""
        status = rel_result.get("status")
        minutes = max(1, self.plugin.db.RELATION_PENDING_TTL // 60)
        if status == "ok":
            p = rel_result["pending"]
            if p.get("direction") == "up":
                lead = f"💞 TA 想和你们的关系更进一步：「{p['from']}」→「{p['to']}」"
            else:
                lead = f"💔 TA 觉得你们之间也许该保持一些距离：「{p['from']}」→「{p['to']}」"
            return (
                f"{lead}\n回复「确认关系」接受，或「取消关系」拒绝（{minutes} 分钟内有效）"
            )
        if status == "already_pending":
            p = rel_result.get("pending") or {}
            return (
                f"⏳ TA 刚才已经提议「{p.get('from', '')}」→「{p.get('to', '')}」，"
                f"先回复「确认关系」或「取消关系」吧（{minutes} 分钟内有效）。"
            )
        return None

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
