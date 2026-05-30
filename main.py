"""
astrbot_plugin_favorability — 好感度系统插件

AI 根据对话内容自主更新用户好感度与评价，支持表情包回应。
基于 AstrBot 框架开发。

架构说明：
  main.py          — 插件入口，注册命令，组装子模块
  models/          — 数据层：FavorabilityManager（好感度 CRUD）
  services/        — 服务层：StickerManager（表情包），PromptManager（提示词）
  llm/             — LLM 层：LLMHandler（请求注入 + 响应解析）
  commands/        — 指令层：UserCommands（用户指令），AdminCommands（管理员指令）
  render/          — 渲染层：FavorabilityRenderer（PIL 图片渲染）
"""

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
    "好感度系统：AI根据对话内容自主更新用户好感度与评价，支持表情包回应，PIL图片渲染",
    "v2.0.0",
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
        try:
            from .render.image import FavorabilityRenderer

            self.renderer = FavorabilityRenderer(data_dir / "render_cache")
            self.has_renderer = True
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

    # ── 管理员指令 ─────────────────────────────────────────

    @filter.command("设置好感度")
    async def cmd_admin_set(self, event: AstrMessageEvent):
        async for r in self.admin_cmds.cmd_admin_set(event):
            yield r

    @filter.command("重置指定好感度")
    async def cmd_admin_reset(self, event: AstrMessageEvent):
        async for r in self.admin_cmds.cmd_admin_reset(event):
            yield r

    # ── 生命周期 ───────────────────────────────────────────

    async def terminate(self):
        pass
