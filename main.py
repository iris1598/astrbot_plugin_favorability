"""
astrbot_plugin_favorability — 好感度系统插件

AI 根据对话内容自主更新用户好感度与评价，支持表情包回应与禁言处罚。
基于 AstrBot 框架开发。

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

from .models.manager import FavorabilityManager
from .services.sticker import StickerManager
from .services.prompt import PromptManager
from .llm.handler import LLMHandler
from .commands.user import UserCommands
from .commands.admin import AdminCommands


@register(
    "astrbot_plugin_favorability",
    "Iris1598",
    "好感度系统：AI根据对话内容自主更新用户好感度与评价，支持表情包回应、禁言处罚，PIL图片渲染",
    "v2.1.0",
)
class FavorabilityPlugin(Star):
    """好感度系统主插件。"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

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

            self.renderer = FavorabilityRenderer(data_dir / "render_cache")
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

    @property
    def favorability_enabled(self) -> bool:
        return bool(self.config.get("favorability_enabled", True))

    @property
    def sticker_enabled(self) -> bool:
        return bool(self.config.get("sticker_enabled", True))

    @property
    def mute_enabled(self) -> bool:
        return bool(self.config.get("mute_enabled", True))

    @property
    def mute_condition(self) -> str:
        return str(self.config.get("mute_condition", "持续恶劣行为（如辱骂、骚扰、刷屏、恶意挑衅）"))

    @property
    def system_time_enabled(self) -> bool:
        return bool(self.config.get("system_time_enabled", True))

    @property
    def user_info_enabled(self) -> bool:
        return bool(self.config.get("user_info_enabled", True))

    def keys(self, event: AstrMessageEvent) -> tuple[str, str]:
        """返回 (group_key, user_id)。"""
        user_id = str(event.get_sender_id())
        group_key = event.unified_msg_origin
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

    @filter.command("好感度排行")
    async def cmd_rank(self, event: AstrMessageEvent):
        async for r in self.user_cmds.cmd_rank(event):
            yield r

    @filter.command("好感度倒序")
    async def cmd_rank_desc(self, event: AstrMessageEvent):
        async for r in self.user_cmds.cmd_rank_desc(event):
            yield r

    @filter.command("重置好感度")
    async def cmd_reset_self(self, event: AstrMessageEvent):
        async for r in self.user_cmds.cmd_reset_self(event):
            yield r

    @filter.command("清理渲染缓存")
    async def cmd_clean_cache(self, event: AstrMessageEvent):
        """手动清理过期的渲染缓存。"""
        if not self.has_renderer:
            yield event.make_result().message("❌ 渲染器未初始化，无法清理缓存。")
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

    # ── 管理员指令 ─────────────────────────────────────────

    @filter.command("设置好感度")
    async def cmd_admin_set(self, event: AstrMessageEvent):
        async for r in self.admin_cmds.cmd_admin_set(event):
            yield r

    @filter.command("重置指定好感度")
    async def cmd_admin_reset(self, event: AstrMessageEvent):
        async for r in self.admin_cmds.cmd_admin_reset(event):
            yield r

    @filter.command("禁言")
    async def cmd_admin_mute(self, event: AstrMessageEvent):
        async for r in self.admin_cmds.cmd_mute(event):
            yield r

    @filter.command("解除禁言")
    async def cmd_admin_unmute(self, event: AstrMessageEvent):
        async for r in self.admin_cmds.cmd_unmute(event):
            yield r

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
