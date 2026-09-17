"""@ 提及解析回归测试。

复现并锁定的问题：
    群聊里用「@机器人 查询好感度」唤醒插件时，会被误解析为查询机器人自己
    的好感度（因为消息链里第一个 At 就是唤醒用的 @机器人）。

覆盖两层：
  1. commands.mentions.extract_at_target_id —— @目标解析本身；
  2. UserCommands.cmd_query —— 查询指令最终查的是谁（真实 bug 场景）。

运行：
  python -m unittest discover -s tests -t . -p "test_at_target.py"
"""

import asyncio
import os
import sys
import types
import unittest
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
# commands 包内部使用相对导入（from ..models.manager import ...），
# 因此必须以插件包的形式导入，需要把插件的上级目录加入 sys.path。
for _p in (str(PLUGIN_ROOT.parent), str(PLUGIN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 优先使用真实 AstrBot（便携版 app 目录），否则注入最小桩
for _cand in [
    os.environ.get("ASTRBOT_APP", ""),
    str(PLUGIN_ROOT.parent / "AstrBot" / "backend" / "app"),
]:
    if _cand and (Path(_cand) / "astrbot").is_dir() and _cand not in sys.path:
        sys.path.insert(0, _cand)

if "astrbot.api" not in sys.modules:

    class _Logger:
        def info(self, message):
            pass

        def warning(self, message):
            pass

        def error(self, message):
            pass

    class _At:
        """与 astrbot.core.message.components.At 保持同名关键字段。"""

        def __init__(self, qq="", name=""):
            self.qq = qq
            self.name = name

    class _AtAll(_At):
        def __init__(self, **_):
            super().__init__(qq="all", name="全体成员")

    class _Plain:
        def __init__(self, text=""):
            self.text = text

    class _Reply:
        """OneBot v11 适配器会给 Reply 组件也带上 qq 字段（兼容用）。"""

        def __init__(self, sender_id="", qq=""):
            self.sender_id = sender_id
            self.qq = qq

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = _Logger()

    components = types.ModuleType("astrbot.api.message_components")
    components.At = _At
    components.AtAll = _AtAll
    components.Plain = _Plain
    components.Reply = _Reply

    event_mod = types.ModuleType("astrbot.api.event")
    event_mod.filter = types.SimpleNamespace()
    event_mod.AstrMessageEvent = type("AstrMessageEvent", (), {})

    api.message_components = components
    api.event = event_mod
    astrbot.api = api
    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.message_components"] = components
    sys.modules["astrbot.api.event"] = event_mod

import astrbot.api.message_components as Comp  # noqa: E402

from astrbot_plugin_favorability.commands.mentions import (  # noqa: E402
    extract_at_target_id,
)
from astrbot_plugin_favorability.commands.user import UserCommands  # noqa: E402

BOT_ID = "10000"  # 机器人自己的 QQ
SENDER_ID = "846370266"  # 发消息的群成员
OTHER_ID = "248690058"  # 被 @ 的另一位群成员
GROUP_KEY = "Iris:GroupMessage:1041386550"


class _FakeEvent:
    """最小事件桩，只实现插件实际用到的方法。"""

    def __init__(
        self,
        components,
        self_id=BOT_ID,
        sender_id=SENDER_ID,
        message_str="查询好感度",
        with_get_messages=True,
    ):
        self._components = list(components)
        self._self_id = self_id
        self._sender_id = sender_id
        self.message_str = message_str
        self.message_obj = types.SimpleNamespace(message=list(components))
        if not with_get_messages:
            # 模拟没有 get_messages() 的旧事件对象
            self.get_messages = None

    def get_self_id(self):
        return self._self_id

    def get_sender_id(self):
        return self._sender_id

    def get_sender_name(self):
        return "测试用户"

    def get_messages(self):
        return list(self._components)

    def plain_result(self, text):
        return text

    def image_result(self, path):
        return path


class _FakePersonaConfig:
    plugin_enabled = True
    favorability_enabled = True


class _FakeDB:
    DEFAULT_RELATION = "普通朋友"

    def __init__(self):
        self.queried = []

    def get_user_info(self, group_key, user_id, persona_id=None):
        self.queried.append(user_id)
        return {
            "score": 0,
            "eval": "",
            "relation": self.DEFAULT_RELATION,
            "pending_rel": None,
        }

    @staticmethod
    def effective_pending(info):
        return None


class _FakePlugin:
    """只提供 UserCommands 需要的接口。"""

    def __init__(self):
        self.db = _FakeDB()
        self.has_renderer = False

    async def resolve_persona_id(self, event, req=None):
        return "default"

    def get_persona_config(self, persona_id):
        return _FakePersonaConfig()

    def keys(self, event):
        return GROUP_KEY, str(event.get_sender_id())


def _run_query(event):
    """执行 /查询好感度，返回被查询的 user_id 列表。"""
    plugin = _FakePlugin()
    cmds = UserCommands(plugin)

    async def _collect():
        return [chunk async for chunk in cmds.cmd_query(event)]

    results = asyncio.run(_collect())
    return plugin, results


class TestExtractAtTargetId(unittest.TestCase):
    def test_at_bot_only_returns_none(self):
        """@机器人 只是唤醒前缀，不应被当成查询目标。"""
        event = _FakeEvent([Comp.At(qq=BOT_ID, name="机器人"), Comp.Plain("查询好感度")])
        self.assertIsNone(extract_at_target_id(event))

    def test_at_bot_and_other_user_returns_other(self):
        """@机器人 唤醒后再 @他人，目标应为他人。"""
        event = _FakeEvent(
            [
                Comp.At(qq=BOT_ID, name="机器人"),
                Comp.Plain(" 查询好感度 "),
                Comp.At(qq=OTHER_ID, name="西格莉卡"),
            ]
        )
        self.assertEqual(extract_at_target_id(event), OTHER_ID)

    def test_plain_at_other_user(self):
        event = _FakeEvent([Comp.At(qq=OTHER_ID, name="西格莉卡")])
        self.assertEqual(extract_at_target_id(event), OTHER_ID)

    def test_at_all_ignored(self):
        """@全体成员 不是用户，必须忽略（含 AtAll 子类）。"""
        self.assertIsNone(extract_at_target_id(_FakeEvent([Comp.AtAll()])))
        self.assertIsNone(
            extract_at_target_id(
                _FakeEvent([Comp.At(qq="all"), Comp.AtAll(), Comp.Plain("查询好感度")])
            )
        )

    def test_reply_component_is_not_an_at(self):
        """Reply 组件同样带 qq 字段，不能被当成 @ 提及。"""
        event = _FakeEvent(
            [Comp.Reply(sender_id=BOT_ID, qq=BOT_ID), Comp.At(qq=BOT_ID), Comp.Plain("查询好感度")]
        )
        self.assertIsNone(extract_at_target_id(event))

    def test_works_without_get_messages(self):
        """回退到 message_obj.message 的旧事件对象同样可用。"""
        event = _FakeEvent(
            [Comp.At(qq=BOT_ID), Comp.At(qq=OTHER_ID)],
            with_get_messages=False,
        )
        self.assertEqual(extract_at_target_id(event), OTHER_ID)

    def test_self_id_unknown_keeps_previous_behaviour(self):
        """拿不到 self_id 时无法识别自身 At，退化为原有行为（取第一个 At）。"""
        event = _FakeEvent([Comp.At(qq=BOT_ID)], self_id="")
        self.assertEqual(extract_at_target_id(event), BOT_ID)


class TestQueryCommandTarget(unittest.TestCase):
    """/查询好感度 最终查的到底是谁。"""

    def test_at_bot_wakeup_queries_sender(self):
        """核心回归：@机器人 查询好感度 必须查发送者自己。"""
        event = _FakeEvent([Comp.At(qq=BOT_ID, name="机器人"), Comp.Plain("查询好感度")])
        plugin, results = _run_query(event)

        self.assertEqual(plugin.db.queried, [SENDER_ID])
        self.assertIn("你的好感度档案", results[0])

    def test_at_bot_then_other_queries_other(self):
        event = _FakeEvent(
            [
                Comp.At(qq=BOT_ID, name="机器人"),
                Comp.Plain(" 查询好感度 "),
                Comp.At(qq=OTHER_ID, name="西格莉卡"),
            ]
        )
        plugin, results = _run_query(event)

        self.assertEqual(plugin.db.queried, [OTHER_ID])
        self.assertIn(OTHER_ID, results[0])

    def test_no_at_queries_sender(self):
        event = _FakeEvent([Comp.Plain("查询好感度")])
        plugin, _ = _run_query(event)
        self.assertEqual(plugin.db.queried, [SENDER_ID])

    def test_text_parameter_still_supported(self):
        """兼容纯文本参数：查询好感度 248690058"""
        event = _FakeEvent(
            [Comp.Plain("查询好感度 248690058")],
            message_str="查询好感度 248690058",
        )
        plugin, _ = _run_query(event)
        self.assertEqual(plugin.db.queried, [OTHER_ID])

    def test_onebot11_at_text_token_supported(self):
        """OneBot v11 会把非首个 @ 渲染成「@昵称(qq)」写进 message_str。"""
        event = _FakeEvent(
            [Comp.At(qq=BOT_ID), Comp.Plain("查询好感度 ")],
            message_str="查询好感度 @西格莉卡(248690058)",
        )
        plugin, _ = _run_query(event)
        self.assertEqual(plugin.db.queried, [OTHER_ID])


if __name__ == "__main__":
    unittest.main()
