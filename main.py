"""
astrbot_plugin_favorability — 好感度 + 关系系统插件

好感度数值只影响 AI 说话的态度；关系档位独立存储，通过 [REL] 提议 +
用户「确认关系」二次确认升降档，每次仅变动相邻一档。
支持表情包回应、禁言处罚与 PIL 图片渲染。基于 AstrBot 框架开发。

架构说明：
  main.py          — 插件入口，注册命令，组装子模块
  models/          — 数据层：FavorabilityManager（好感度 CRUD、禁言管理）
  services/        — 服务层：StickerManager（表情包），PromptManager（提示词）
  llm/             — LLM 层：LLMHandler（请求注入 + 禁言拦截 + 响应解析）
  commands/        — 指令层：UserCommands（用户指令），AdminCommands（管理员指令）
  render/          — 渲染层：FavorabilityRenderer（PIL 图片渲染）
"""

import asyncio
import time

import astrbot.api.message_components as Comp
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api import AstrBotConfig, logger

from .models.manager import FavorabilityManager, group_storage_key
from .services.sticker import StickerManager
from .services.prompt import (
    DEFAULT_INTERACTION_HINT,
    DEFAULT_STICKER_CONDITION,
    FAVORABILITY_PROMPT_DEFAULTS,
    PROMPT_PRESET_NAMES,
    RELATION_KEY_MAP,
    PromptManager,
)
from .llm.handler import LLMHandler
from .commands.user import UserCommands
from .commands.admin import AdminCommands


@register(
    "astrbot_plugin_favorability",
    "Iris1598",
    "好感度系统：好感度数值影响说话态度，关系档位独立管理（提议+确认升降档），"
    "支持表情包回应、禁言处罚，PIL图片渲染",
    "v3.0.0",
)
class FavorabilityPlugin(Star):
    _SETTING_GROUPS = {
        "plugin_enabled": "feature_settings",
        "favorability_enabled": "feature_settings",
        "sticker_enabled": "feature_settings",
        "mute_enabled": "feature_settings",
        "relation_enabled": "feature_settings",
        "prompt_preset": "prompt_settings",
        "mute_condition": "prompt_settings",
        "sticker_condition": "prompt_settings",
        "favorability_prompt_core": "prompt_settings",
        "favorability_prompt_behavior": "prompt_settings",
        "favorability_prompt_relation": "prompt_settings",
        "favorability_prompt_mute": "prompt_settings",
        "favorability_prompt_security": "prompt_settings",
        "relation_guideline_lover": "prompt_settings",
        "relation_guideline_confidant": "prompt_settings",
        "relation_guideline_friend": "prompt_settings",
        "relation_guideline_acquaintance": "prompt_settings",
        "relation_guideline_estranged": "prompt_settings",
        "relation_guideline_rival": "prompt_settings",
        "relation_guideline_severed": "prompt_settings",
        "persona_overrides": "persona_settings",
        "interaction_hint_enabled": "feature_settings",
        "interaction_hint_text": "prompt_settings",
        "system_time_enabled": "context_settings",
        "user_info_enabled": "context_settings",
        "render_theme": "render_settings",
    }
    _CONFIG_LAYOUT_VERSION = 4

    """好感度系统主插件。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._migrate_grouped_config()
        self._restore_empty_prompt_defaults()

        # ── 数据层 ──
        data_dir = StarTools.get_data_dir()
        self.db = FavorabilityManager(data_dir)
        self.stickers = StickerManager(data_dir / "stickers")

        # ── 服务层（PromptManager 为纯静态方法，无需实例化） ──
        self.prompt = PromptManager  # 方便引用

        # ── LLM 层 ──
        self.llm_handler = LLMHandler(self)

        # ── 指令层 ──
        self.user_cmds = UserCommands(self)
        self.admin_cmds = AdminCommands(self)

        # ── 渲染层 ──
        self._cache_cleanup_task: asyncio.Task | None = None
        try:
            from .render.image import FavorabilityRenderer

            self.renderer = FavorabilityRenderer(
                data_dir / "render_cache",
                theme=str(self._get_setting("render_theme", "dark")),
            )
            self.has_renderer = True
            # 启动时清理过期缓存
            deleted, remaining = self.renderer.cleanup_cache()
            if deleted or remaining:
                logger.info(
                    f"[favorability] 渲染缓存启动清理: 删除了 {deleted} 个过期文件，"
                    f"剩余 {remaining} 个"
                )
            # 启动定时清理后台任务
            self._start_cache_cleanup_task()
        except Exception as e:
            self.renderer = None
            self.has_renderer = False
            logger.warning(f"[favorability] PIL 渲染器初始化失败，将使用文本模式: {e}")

    # ── 配置属性 ────────────────────────────────────────────

    def _get_setting(self, key: str, default=None):
        """读取分组配置，并兼容旧版平铺配置。"""
        group_name = self._SETTING_GROUPS.get(key)
        if group_name:
            group = self.config.get(group_name)
            if isinstance(group, dict) and key in group:
                return group.get(key)
        return self.config.get(key, default)

    def _set_setting(self, key: str, value) -> None:
        group_name = self._SETTING_GROUPS.get(key)
        if not group_name:
            self.config[key] = value
            return
        group = self.config.get(group_name)
        if not isinstance(group, dict):
            group = {}
            self.config[group_name] = group
        group[key] = value

    def _migrate_grouped_config(self) -> None:
        """把旧版平铺配置迁移到新的设置分组，避免升级后丢失配置。"""
        try:
            layout_version = int(self.config.get("_config_layout_version", 0) or 0)
        except (TypeError, ValueError):
            layout_version = 0

        if layout_version >= self._CONFIG_LAYOUT_VERSION:
            return

        for key, group_name in self._SETTING_GROUPS.items():
            group = self.config.get(group_name)
            if not isinstance(group, dict):
                group = {}
                self.config[group_name] = group
            if key in self.config:
                group[key] = self.config[key]

        self.config["_config_layout_version"] = self._CONFIG_LAYOUT_VERSION
        save_config = getattr(self.config, "save_config", None)
        if callable(save_config):
            try:
                save_config()
            except Exception as e:
                logger.warning(f"[favorability] 保存分组配置失败: {e}")
        logger.info("[favorability] 已将旧版平铺配置迁移到新的设置分组")

    def _restore_empty_prompt_defaults(self):
        """将留空的提示词配置恢复为内置默认值并持久化。"""
        restored = []
        prompt_defaults = {
            **FAVORABILITY_PROMPT_DEFAULTS,
            "sticker_condition": DEFAULT_STICKER_CONDITION,
            "interaction_hint_text": DEFAULT_INTERACTION_HINT,
        }
        for key, default in prompt_defaults.items():
            value = self._get_setting(key, "")
            if not str(value or "").strip():
                self._set_setting(key, default)
                restored.append(key)

        if not restored:
            return

        save_config = getattr(self.config, "save_config", None)
        if callable(save_config):
            try:
                save_config()
            except Exception as e:
                logger.warning(f"[favorability] 保存默认提示词配置失败: {e}")
        logger.info(
            f"[favorability] 已恢复 {len(restored)} 项留空的默认提示词配置"
        )

    @property
    def plugin_enabled(self) -> bool:
        return bool(self._get_setting("plugin_enabled", True))

    @property
    def favorability_enabled(self) -> bool:
        return bool(self._get_setting("favorability_enabled", True))

    @property
    def sticker_enabled(self) -> bool:
        return bool(self._get_setting("sticker_enabled", True))

    @property
    def sticker_condition(self) -> str:
        return str(self._get_setting("sticker_condition", "") or "")

    @property
    def prompt_preset(self) -> str:
        value = str(self._get_setting("prompt_preset", "default") or "").strip().lower()
        return value if value in PROMPT_PRESET_NAMES else "default"

    @property
    def mute_enabled(self) -> bool:
        return bool(self._get_setting("mute_enabled", True))

    @property
    def relation_enabled(self) -> bool:
        return bool(self._get_setting("relation_enabled", True))

    @property
    def favorability_prompt_relation(self) -> str:
        return str(self._get_setting("favorability_prompt_relation", "") or "")

    @property
    def mute_condition(self) -> str:
        return str(self._get_setting("mute_condition", "持续恶劣行为（如辱骂、骚扰、刷屏、恶意挑衅）"))

    @property
    def favorability_prompt_core(self) -> str:
        return str(self._get_setting("favorability_prompt_core", "") or "")

    @property
    def favorability_prompt_behavior(self) -> str:
        return str(self._get_setting("favorability_prompt_behavior", "") or "")

    @property
    def favorability_prompt_mute(self) -> str:
        return str(self._get_setting("favorability_prompt_mute", "") or "")

    @property
    def favorability_prompt_security(self) -> str:
        return str(self._get_setting("favorability_prompt_security", "") or "")

    @property
    def system_time_enabled(self) -> bool:
        return bool(self._get_setting("system_time_enabled", True))

    @property
    def user_info_enabled(self) -> bool:
        return bool(self._get_setting("user_info_enabled", True))

    @property
    def interaction_hint_enabled(self) -> bool:
        return bool(self._get_setting("interaction_hint_enabled", True))

    @property
    def interaction_hint_text(self) -> str:
        return str(
            self._get_setting("interaction_hint_text", DEFAULT_INTERACTION_HINT)
            or DEFAULT_INTERACTION_HINT
        )

    @property
    def relation_guidelines(self) -> dict[str, str]:
        """全局配置中自定义的七档关系行为准则字典。"""
        guidelines = {}
        for key, name in RELATION_KEY_MAP.items():
            val = self._get_setting(key, "")
            if val and str(val).strip():
                guidelines[name] = str(val).strip()
        return guidelines

    def get_persona_config(self, persona_id: str) -> "PersonaConfig":
        """获取指定人格的专属配置视图（未配置项自动回退全局默认配置）。"""
        return PersonaConfig(self, persona_id)

    async def resolve_persona_id(
        self, event: AstrMessageEvent, req: ProviderRequest = None
    ) -> str:
        """多层级智能解析当前会话生效的人格 ID / 名称，未识别时回退为 'default'。"""
        # 1. 优先尝试从 req.conversation 获取
        if req is not None:
            conv = getattr(req, "conversation", None)
            if conv:
                pid = getattr(conv, "persona_id", None)
                if pid and str(pid).strip() and str(pid).strip() != "[%None]":
                    return str(pid).strip()

        # 2. 尝试通过 AstrBot context.persona_manager 解析
        pm = getattr(self.context, "persona_manager", None)
        if pm and hasattr(pm, "resolve_selected_persona"):
            try:
                conv_pid = None
                if req is not None and getattr(req, "conversation", None):
                    conv_pid = getattr(req.conversation, "persona_id", None)
                elif hasattr(self.context, "conversation_manager"):
                    curr_cid = await self.context.conversation_manager.get_curr_conversation_id(
                        event.unified_msg_origin
                    )
                    if curr_cid:
                        conv = await self.context.conversation_manager.get_conversation(
                            event.unified_msg_origin, curr_cid
                        )
                        if conv:
                            conv_pid = getattr(conv, "persona_id", None)

                cfg = (
                    getattr(self.context, "get_config", lambda umo=None: {})(
                        umo=event.unified_msg_origin
                    )
                    or {}
                )
                res = await pm.resolve_selected_persona(
                    umo=event.unified_msg_origin,
                    conversation_persona_id=conv_pid,
                    platform_name=event.get_platform_name()
                    if hasattr(event, "get_platform_name")
                    else "",
                    provider_settings=cfg,
                )
                if res and res[0] and str(res[0]).strip() and str(res[0]).strip() != "[%None]":
                    return str(res[0]).strip()
            except Exception as e:
                logger.debug(f"[favorability] 解析 persona 异常: {e}")

        # 3. 尝试通过 conversation_manager 获取当前 conversation 的 persona_id
        cm = getattr(self.context, "conversation_manager", None)
        if cm and hasattr(cm, "get_curr_conversation_id"):
            try:
                curr_cid = await cm.get_curr_conversation_id(event.unified_msg_origin)
                if curr_cid:
                    conv = await cm.get_conversation(
                        event.unified_msg_origin, curr_cid
                    )
                    if conv and getattr(conv, "persona_id", None):
                        pid = str(conv.persona_id).strip()
                        if pid and pid != "[%None]":
                            return pid
            except Exception as e:
                logger.debug(f"[favorability] 从 conversation_manager 获取 persona 异常: {e}")

        return "default"

    def keys(self, event: AstrMessageEvent) -> tuple[str, str]:
        """返回 (group_key, user_id)。

        官方会话隔离(unique_session)开启后群聊 UMO 变为每人一个
        {平台}:GroupMessage:{用户}_{群}，而好感度（排行/禁言/互相查询）
        必须按群共享存储，这里统一归一化回群级键；私聊/webchat 不受影响。
        """
        user_id = str(event.get_sender_id())
        group_key = group_storage_key(event.unified_msg_origin, user_id)
        return group_key, user_id

    # ── LLM 事件钩子 ───────────────────────────────────────

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        await self.llm_handler.on_llm_request(event, req)

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        await self.llm_handler.on_llm_response(event, resp)

    # ── 渲染缓存定时清理 ───────────────────────────────────

    CACHE_CLEANUP_INTERVAL = 3600  # 默认每小时清理一次

    def _start_cache_cleanup_task(self):
        """启动定时缓存清理后台任务。"""
        if not self.has_renderer:
            return

        async def _cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.CACHE_CLEANUP_INTERVAL)
                    deleted, remaining = self.renderer.cleanup_cache()
                    if deleted:
                        logger.info(
                            f"[favorability] 定时清理渲染缓存: 删除了 {deleted} 个过期文件，"
                            f"剩余 {remaining} 个"
                        )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"[favorability] 缓存清理异常: {e}")

        self._cache_cleanup_task = asyncio.create_task(_cleanup_loop())

    # ── 用户指令 ───────────────────────────────────────────

    @filter.command("查询好感度")
    async def cmd_query(self, event: AstrMessageEvent):
        async for r in self.user_cmds.cmd_query(event):
            yield r
        event.stop_event()

    @filter.command("好感度排行")
    async def cmd_rank(self, event: AstrMessageEvent):
        async for r in self.user_cmds.cmd_rank(event):
            yield r
        event.stop_event()

    @filter.command("好感度倒序")
    async def cmd_rank_desc(self, event: AstrMessageEvent):
        async for r in self.user_cmds.cmd_rank_desc(event):
            yield r
        event.stop_event()

    @filter.command("重置好感度")
    async def cmd_reset_self(self, event: AstrMessageEvent):
        async for r in self.user_cmds.cmd_reset_self(event):
            yield r
        event.stop_event()

    @filter.command("确认关系")
    async def cmd_confirm_relation(self, event: AstrMessageEvent):
        async for r in self.user_cmds.cmd_confirm_relation(event):
            yield r
        event.stop_event()

    @filter.command("取消关系")
    async def cmd_cancel_relation(self, event: AstrMessageEvent):
        async for r in self.user_cmds.cmd_cancel_relation(event):
            yield r
        event.stop_event()

    @filter.command("清理渲染缓存")
    async def cmd_clean_cache(self, event: AstrMessageEvent):
        """手动清理过期的渲染缓存。"""
        if not self.has_renderer:
            yield event.make_result().message("❌ 渲染器未初始化，无法清理缓存。")
            event.stop_event()
            return

        info = self.renderer.get_cache_info()
        deleted, remaining = self.renderer.cleanup_cache(max_age=0)  # 全部清理
        size_mb = info["size_bytes"] / 1024 / 1024
        yield event.make_result().message(
            f"✅ 已清理渲染缓存\n"
            f"• 清理前: {info['count']} 个文件 ({size_mb:.1f} MB)\n"
            f"• 已删除: {deleted} 个文件\n"
            f"• 剩余: {remaining} 个文件\n"
            f"• 缓存目录: {info['dir']}"
        )
        event.stop_event()

    # ── 管理员指令 ─────────────────────────────────────────

    @filter.command("设置好感度")
    async def cmd_admin_set(self, event: AstrMessageEvent):
        async for r in self.admin_cmds.cmd_admin_set(event):
            yield r
        event.stop_event()

    @filter.command("重置指定好感度")
    async def cmd_admin_reset(self, event: AstrMessageEvent):
        async for r in self.admin_cmds.cmd_admin_reset(event):
            yield r
        event.stop_event()

    @filter.command("设置关系")
    async def cmd_admin_set_relation(self, event: AstrMessageEvent):
        async for r in self.admin_cmds.cmd_admin_set_relation(event):
            yield r
        event.stop_event()

    @filter.command("禁言")
    async def cmd_admin_mute(self, event: AstrMessageEvent):
        async for r in self.admin_cmds.cmd_mute(event):
            yield r
        event.stop_event()

    @filter.command("解除禁言")
    async def cmd_admin_unmute(self, event: AstrMessageEvent):
        async for r in self.admin_cmds.cmd_unmute(event):
            yield r
        event.stop_event()

    # ── 生命周期 ───────────────────────────────────────────

    async def terminate(self):
        """插件卸载时取消定时清理任务。"""
        if self._cache_cleanup_task is not None:
            self._cache_cleanup_task.cancel()
            try:
                await self._cache_cleanup_task
            except asyncio.CancelledError:
                pass
            self._cache_cleanup_task = None
            logger.info("[favorability] 渲染缓存定时清理任务已取消")


class PersonaConfig:
    """针对特定人格的配置视图，支持独立覆盖与回退全局默认配置。"""

    def __init__(self, plugin: FavorabilityPlugin, persona_id: str):
        self.plugin = plugin
        self.persona_id = (persona_id or "").strip() or "default"
        self.override = self._find_override()

    def _find_override(self) -> dict:
        overrides = self.plugin._get_setting("persona_overrides", [])
        if not isinstance(overrides, list):
            return {}
        for item in overrides:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("persona_id") or "").strip()
            if pid and pid == self.persona_id:
                return item
        return {}

    def get(self, key: str, default=None):
        if self.override and key in self.override:
            val = self.override[key]
            if isinstance(val, str):
                if val.strip():
                    return val
            elif val is not None:
                return val
        return self.plugin._get_setting(key, default)

    @property
    def plugin_enabled(self) -> bool:
        return bool(self.get("plugin_enabled", True))

    @property
    def favorability_enabled(self) -> bool:
        return bool(self.get("favorability_enabled", True))

    @property
    def sticker_enabled(self) -> bool:
        return bool(self.get("sticker_enabled", True))

    @property
    def mute_enabled(self) -> bool:
        return bool(self.get("mute_enabled", True))

    @property
    def relation_enabled(self) -> bool:
        return bool(self.get("relation_enabled", True))

    @property
    def sticker_dir_name(self) -> str:
        val = str(self.get("sticker_dir_name", "") or "").strip()
        return val or self.persona_id

    @property
    def prompt_preset(self) -> str:
        value = str(self.get("prompt_preset", "default") or "").strip().lower()
        return value if value in PROMPT_PRESET_NAMES else "default"

    @property
    def sticker_condition(self) -> str:
        return str(self.get("sticker_condition", "") or "")

    @property
    def mute_condition(self) -> str:
        return str(
            self.get(
                "mute_condition", "持续恶劣行为（如辱骂、骚扰、刷屏、恶意挑衅）"
            )
            or ""
        )

    @property
    def favorability_prompt_core(self) -> str:
        return str(self.get("favorability_prompt_core", "") or "")

    @property
    def favorability_prompt_behavior(self) -> str:
        return str(self.get("favorability_prompt_behavior", "") or "")

    @property
    def favorability_prompt_relation(self) -> str:
        return str(self.get("favorability_prompt_relation", "") or "")

    @property
    def favorability_prompt_mute(self) -> str:
        return str(self.get("favorability_prompt_mute", "") or "")

    @property
    def favorability_prompt_security(self) -> str:
        return str(self.get("favorability_prompt_security", "") or "")

    @property
    def relation_guidelines(self) -> dict[str, str]:
        """优先使用人格专属七档关系态度，未填写项回退全局设置。"""
        guidelines = {}
        for key, name in RELATION_KEY_MAP.items():
            val = self.get(key, "")
            if val and str(val).strip():
                guidelines[name] = str(val).strip()
        return guidelines

