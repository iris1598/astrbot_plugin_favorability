import asyncio
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path
import unittest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

if "astrbot.api" not in sys.modules:
    class _Logger:
        def info(self, message):
            pass

        def warning(self, message):
            pass

        def error(self, message):
            pass

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = _Logger()
    astrbot.api = api
    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api

from models.manager import FavorabilityManager
from services.prompt import (
    DEFAULT_RELATION_GUIDELINES,
    PromptManager,
    build_relation_guideline,
    resolve_relation_guidelines,
)
from services.sticker import StickerManager


class TestPersonaDatabaseIsolation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        self.manager = FavorabilityManager(self.data_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_persona_stores_in_root_json(self):
        """测试默认人格直接存储在 data_dir / favorability.json，保持零迁移兼容"""
        asyncio.run(
            self.manager.set_score("grp1", "user1", 80, persona_id="default")
        )
        default_file = self.data_dir / "favorability.json"
        self.assertTrue(default_file.exists())

        data = json.loads(default_file.read_text(encoding="utf-8"))
        self.assertIn("grp1", data)
        self.assertIn("user1", data["grp1"])
        self.assertEqual(data["grp1"]["user1"]["score"], 80)

    def test_custom_persona_stores_in_subfolder(self):
        """测试非默认人格存储在 personas/{persona_id}/favorability.json"""
        asyncio.run(
            self.manager.set_score("grp1", "user1", 30, persona_id="tsundere")
        )
        tsundere_file = self.data_dir / "personas" / "tsundere" / "favorability.json"
        self.assertTrue(tsundere_file.exists())

        # 验证默认人格文件未被污染
        default_data = json.loads((self.data_dir / "favorability.json").read_text(encoding="utf-8"))
        self.assertNotIn("grp1", default_data)

        # 验证读取
        record = self.manager.get_user_info("grp1", "user1", persona_id="tsundere")
        self.assertEqual(record["score"], 30)

    def test_multiple_personas_isolation(self):
        """测试多个人格之间好感度完全隔离互不影响"""
        asyncio.run(self.manager.set_score("grp1", "alice", 10, persona_id="default"))
        asyncio.run(self.manager.set_score("grp1", "alice", 99, persona_id="waifu"))
        asyncio.run(self.manager.set_score("grp1", "alice", -50, persona_id="villain"))

        self.assertEqual(self.manager.get_user_info("grp1", "alice", persona_id="default")["score"], 10)
        self.assertEqual(self.manager.get_user_info("grp1", "alice", persona_id="waifu")["score"], 99)
        self.assertEqual(self.manager.get_user_info("grp1", "alice", persona_id="villain")["score"], -50)

        # 验证关系提议也是隔离的
        asyncio.run(self.manager.propose_relation("grp1", "alice", "up", persona_id="waifu"))
        waifu_pending = self.manager.get_user_info("grp1", "alice", persona_id="waifu")["pending_rel"]
        self.assertIsNotNone(waifu_pending)
        self.assertEqual(waifu_pending["to"], "熟络好友")
        default_pending = self.manager.get_user_info("grp1", "alice", persona_id="default")["pending_rel"]
        self.assertIsNone(default_pending)
        villain_pending = self.manager.get_user_info("grp1", "alice", persona_id="villain")["pending_rel"]
        self.assertIsNone(villain_pending)

        # 验证静音状态也是隔离的
        asyncio.run(self.manager.mute_user("grp1", "alice", 60, persona_id="villain"))
        self.assertTrue(self.manager.is_muted("grp1", "alice", persona_id="villain"))
        self.assertFalse(self.manager.is_muted("grp1", "alice", persona_id="default"))
        self.assertFalse(self.manager.is_muted("grp1", "alice", persona_id="waifu"))

    def test_ranking_is_isolated(self):
        """测试排行榜在各个人格下独立"""
        asyncio.run(self.manager.set_score("grp1", "user_a", 100, persona_id="persona1"))
        asyncio.run(self.manager.set_score("grp1", "user_b", 50, persona_id="persona1"))

        asyncio.run(self.manager.set_score("grp1", "user_a", 20, persona_id="persona2"))
        asyncio.run(self.manager.set_score("grp1", "user_b", 80, persona_id="persona2"))

        rank1 = self.manager.get_ranked_users("grp1", top_n=10, persona_id="persona1")
        self.assertEqual(rank1[0][0], "user_a")
        self.assertEqual(rank1[1][0], "user_b")

        rank2 = self.manager.get_ranked_users("grp1", top_n=10, persona_id="persona2")
        self.assertEqual(rank2[0][0], "user_b")
        self.assertEqual(rank2[1][0], "user_a")


class TestPersonaStickerIsolation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.stickers_dir = Path(self.temp_dir) / "stickers"
        self.stickers_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sticker_fallback_to_root_when_no_subfolder(self):
        """当没有专属子目录时回退到根目录"""
        cat_dir = self.stickers_dir / "happy"
        cat_dir.mkdir()
        (cat_dir / "1.jpg").write_text("dummy")

        sticker_mgr = StickerManager(self.stickers_dir)
        # default 人格
        cats_default = sticker_mgr.get_categories(persona_id="default")
        self.assertIn("happy", cats_default)

        # 未建立子目录的人格回退到根目录
        cats_other = sticker_mgr.get_categories(persona_id="other_persona")
        self.assertIn("happy", cats_other)

    def test_sticker_persona_subfolder_isolation(self):
        """当存在人格专属子目录时，优先使用专属表情包"""
        # default 人格专属目录
        default_dir = self.stickers_dir / "default" / "happy"
        default_dir.mkdir(parents=True)
        (default_dir / "normal.jpg").write_text("dummy")

        # 人格 tsundere 专属目录
        tsundere_dir = self.stickers_dir / "tsundere" / "angry"
        tsundere_dir.mkdir(parents=True)
        (tsundere_dir / "tsun.jpg").write_text("dummy")

        sticker_mgr = StickerManager(self.stickers_dir)

        # tsundere 应该看到 angry，而没有 happy
        tsundere_cats = sticker_mgr.get_categories(persona_id="tsundere")
        self.assertEqual(tsundere_cats, ["angry"])

        # default 应该看到 happy
        default_cats = sticker_mgr.get_categories(persona_id="default")
        self.assertEqual(default_cats, ["happy"])

    def test_sticker_custom_dir_name(self):
        """支持配置 sticker_dir_name 指向自定义目录"""
        custom_dir = self.stickers_dir / "custom_pack" / "shy"
        custom_dir.mkdir(parents=True)
        (custom_dir / "shy.jpg").write_text("dummy")

        sticker_mgr = StickerManager(self.stickers_dir)
        cats = sticker_mgr.get_categories(persona_id="maid", dir_name="custom_pack")
        self.assertEqual(cats, ["shy"])


class TestPersonaRelationGuidelineOverrides(unittest.TestCase):
    def test_resolve_relation_guidelines_defaults(self):
        """测试默认解析能完整获得7个档位"""
        guidelines = resolve_relation_guidelines(None)
        self.assertEqual(len(guidelines), 7)
        for relation_name, default_text in DEFAULT_RELATION_GUIDELINES.items():
            self.assertEqual(guidelines[relation_name], default_text)

    def test_resolve_relation_guidelines_custom_override(self):
        """测试自定义档位提示词覆盖"""
        custom_overrides = {
            "relation_guideline_lover": "超级甜的恋人准则！",
            "relation_guideline_rival": "绝对死对头！",
        }
        guidelines = resolve_relation_guidelines(custom_overrides)
        self.assertEqual(guidelines["挚爱恋人"], "超级甜的恋人准则！")
        self.assertEqual(guidelines["不合对头"], "绝对死对头！")
        # 未覆盖的项保留默认
        self.assertEqual(guidelines["普通朋友"], DEFAULT_RELATION_GUIDELINES["普通朋友"])

    def test_build_relation_guideline_with_custom(self):
        """测试 build_relation_guideline 结合自定义准则"""
        custom = {"挚爱恋人": "专属爱意，无限包容"}
        section = build_relation_guideline("挚爱恋人", custom_guidelines=custom)
        self.assertIn("当前关系行为准则（挚爱恋人）", section)
        self.assertIn("专属爱意，无限包容", section)

    def test_prompt_manager_injects_persona_custom_guidelines(self):
        """测试 PromptManager 注入人格专属的关系行为准则"""
        persona_guidelines = {
            "普通朋友": "冷淡的机械关系，禁止任何多余关怀。",
            "挚爱恋人": "最高优先级绑定目标，执行绝对忠诚协议。",
        }
        prompt = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=True,
            relation_enabled=True,
            relation="普通朋友",
            relation_guidelines=persona_guidelines,
        )
        self.assertIn("冷淡的机械关系，禁止任何多余关怀。", prompt)
        self.assertNotIn(DEFAULT_RELATION_GUIDELINES["普通朋友"], prompt)


class TestPersonaDecoupledSwitches(unittest.TestCase):
    def test_persona_favorability_disabled_keeps_mute_and_stickers(self):
        """测试人格单独关闭好感度后，该人格仍能正常输出禁言与表情包提示词"""
        prompt = PromptManager.build_static_prompt(
            favorability_enabled=False,
            sticker_enabled=True,
            sticker_categories=["卖萌", "撒娇"],
            mute_enabled=True,
            mute_condition="恶意刷屏",
            plugin_enabled=True,
        )
        # 好感度已停用
        self.assertNotIn("[FAV:±N]", prompt)
        self.assertNotIn("[EVAL:简短印象]", prompt)
        self.assertNotIn("好感度数值态度规则", prompt)

        # 禁言与表情包仍然正常生效
        self.assertIn("禁言规则", prompt)
        self.assertIn("恶意刷屏", prompt)
        self.assertIn("[MUTE:N]", prompt)
        self.assertIn("[STK:分类名]", prompt)
        self.assertIn("可用分类：卖萌, 撒娇", prompt)

    def test_persona_plugin_disabled_turns_off_everything(self):
        """测试人格关闭插件总开关后，提示词完全为空"""
        prompt = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=True,
            mute_enabled=True,
            plugin_enabled=False,
        )
        self.assertEqual(prompt, "")


if __name__ == "__main__":
    unittest.main()
