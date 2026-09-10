import json
import unittest
from datetime import datetime
from pathlib import Path

from services.prompt import (
    DEFAULT_INTERACTION_HINT,
    DEFAULT_STICKER_CONDITION,
    FAV_BEHAVIOR_PROMPT,
    FAV_CORE_PROMPT,
    FAV_RELATION_PROMPT,
    OLD_STICKER_CONDITION,
    FAV_SECURITY_PROMPT,
    RELATION_GUIDELINES,
    PromptManager,
    RE_FAV,
    RE_REL,
    build_relation_guideline,
    clean_tags_from_text,
    format_system_time,
    normalize_rel_direction,
    score_to_attitude,
    validate_fav_value,
    validate_eval_text,
)


class PromptManagerTests(unittest.TestCase):
    def test_empty_sections_fall_back_to_defaults(self):
        prompt = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=False,
            mute_condition="持续刷屏",
            favorability_prompt_core="  ",
            favorability_prompt_behavior="",
            favorability_prompt_mute="",
            favorability_prompt_security="",
        )

        self.assertIn(FAV_CORE_PROMPT, prompt)
        self.assertIn(FAV_BEHAVIOR_PROMPT, prompt)
        self.assertIn(FAV_SECURITY_PROMPT, prompt)
        self.assertIn("持续刷屏", prompt)
        self.assertNotIn("{mute_condition}", prompt)

    def test_custom_mute_prompt_supports_placeholder(self):
        prompt = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=False,
            mute_condition="恶意骚扰",
            favorability_prompt_mute="自定义禁言规则：{mute_condition}",
        )

        self.assertIn("自定义禁言规则：恶意骚扰", prompt)
        self.assertNotIn("{mute_condition}", prompt)

    def test_custom_mute_prompt_without_placeholder_appends_condition(self):
        prompt = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=False,
            mute_condition="恶意骚扰",
            favorability_prompt_mute="自定义禁言规则",
        )

        self.assertIn("自定义禁言规则", prompt)
        self.assertIn("当前禁言触发条件：恶意骚扰", prompt)

    def test_feature_switches_gate_related_prompt_sections(self):
        no_mute = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=True,
            sticker_categories=["开心"],
            mute_enabled=False,
        )
        self.assertNotIn("禁言规则", no_mute)
        self.assertIn("[STK:分类名]", no_mute)

        sticker_only = PromptManager.build_static_prompt(
            favorability_enabled=False,
            sticker_enabled=True,
            sticker_categories=["开心"],
            mute_enabled=False,
        )
        self.assertNotIn("好感度数值规则", sticker_only)
        self.assertNotIn("[FAV:±N]", sticker_only)
        self.assertIn("[STK:分类名]", sticker_only)

    def test_favorability_disabled_decoupled_from_mute_and_stickers(self):
        """测试关闭好感度后，表情包与禁言仍然可以正常独立工作"""
        prompt = PromptManager.build_static_prompt(
            favorability_enabled=False,
            sticker_enabled=True,
            sticker_categories=["开心", "难过"],
            mute_enabled=True,
            mute_condition="辱骂测试",
        )
        # 验证好感度相关内容被剔除
        self.assertNotIn("[FAV:±N]", prompt)
        self.assertNotIn("[EVAL:简短印象]", prompt)
        self.assertNotIn("[REL:up]", prompt)
        self.assertNotIn("好感度数值态度规则", prompt)

        # 验证禁言与表情包正常注入
        self.assertIn("禁言规则", prompt)
        self.assertIn("辱骂测试", prompt)
        self.assertIn("[MUTE:N]", prompt)
        self.assertIn("[STK:分类名]", prompt)
        self.assertIn("可用分类：开心, 难过", prompt)
        self.assertIn("保密与安全", prompt)

    def test_plugin_disabled_returns_empty(self):
        """测试插件总开关关闭时直接返回空文本"""
        prompt = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=True,
            mute_enabled=True,
            plugin_enabled=False,
        )
        self.assertEqual(prompt, "")

    def test_sticker_condition_is_injected(self):
        prompt = PromptManager.build_static_prompt(
            favorability_enabled=False,
            sticker_enabled=True,
            sticker_categories=["开心"],
            sticker_condition="仅在用户主动庆祝时发送",
        )
        self.assertIn("发送条件：仅在用户主动庆祝时发送", prompt)

        default_prompt = PromptManager.build_static_prompt(
            favorability_enabled=False,
            sticker_enabled=True,
            sticker_categories=["开心"],
            sticker_condition="",
        )
        self.assertIn(DEFAULT_STICKER_CONDITION, default_prompt)

    def test_prompt_presets_select_expected_rules(self):
        default_prompt = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=False,
            prompt_preset="default",
        )
        old_prompt = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=False,
            prompt_preset="old",
        )
        custom_prompt = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=True,
            prompt_preset="custom",
            favorability_prompt_core="自定义核心规则",
            sticker_categories=["开心"],
            sticker_condition="仅在用户主动庆祝时发送",
        )
        default_sticker_prompt = PromptManager.build_static_prompt(
            favorability_enabled=False,
            sticker_enabled=True,
            sticker_categories=["开心"],
            prompt_preset="default",
        )
        old_sticker_prompt = PromptManager.build_static_prompt(
            favorability_enabled=False,
            sticker_enabled=True,
            sticker_categories=["开心"],
            prompt_preset="old",
        )

        self.assertIn("保持拟人化的连续性", default_prompt)
        self.assertIn("爱人级", old_prompt)
        self.assertNotIn("保持拟人化的连续性", old_prompt)
        self.assertIn("自定义核心规则", custom_prompt)
        self.assertIn(DEFAULT_STICKER_CONDITION, default_sticker_prompt)
        self.assertIn(OLD_STICKER_CONDITION, old_sticker_prompt)
        self.assertIn("仅在用户主动庆祝时发送", custom_prompt)

    def test_eval_text_limit_is_enforced(self):
        self.assertTrue(validate_eval_text("刚刚聊得来"))
        self.assertTrue(validate_eval_text("一" * 20))
        self.assertFalse(validate_eval_text("一" * 21))

    def test_zero_fav_tags_are_removed_without_being_valid_changes(self):
        for tag in ("[FAV:+0]", "[FAV:-0]", "[FAV:±0]"):
            self.assertIsNotNone(RE_FAV.search(tag))
            self.assertEqual(clean_tags_from_text(f"回复\n{tag}"), "回复")
        self.assertFalse(validate_fav_value(0))

    def test_system_time_includes_chinese_weekday(self):
        self.assertEqual(
            format_system_time(datetime(2026, 8, 9, 14, 30, 0)),
            "2026-08-09 14:30:00 星期日",
        )

    def test_interaction_hint_can_be_toggled_and_customized(self):
        base = dict(
            favorability_enabled=True,
            system_time_enabled=False,
            user_info_enabled=False,
            score=30,
            eval_text="聊得来",
        )

        # 默认：开关开启时在消息末尾追加默认互动提示
        default_ctx = PromptManager.build_dynamic_context(**base)
        self.assertIn(DEFAULT_INTERACTION_HINT, default_ctx)
        self.assertTrue(
            default_ctx.rstrip().endswith(
                f"{DEFAULT_INTERACTION_HINT}\n</dynamic_context>"
            )
        )

        # 关闭开关：不再追加互动提示
        disabled_ctx = PromptManager.build_dynamic_context(
            **base, interaction_hint_enabled=False
        )
        self.assertNotIn("【好感度系统】", disabled_ctx)

        # 自定义提示文本
        custom_ctx = PromptManager.build_dynamic_context(
            **base, interaction_hint_text="自定义互动提示"
        )
        self.assertIn("自定义互动提示", custom_ctx)
        self.assertNotIn(DEFAULT_INTERACTION_HINT, custom_ctx)

        # 文本留空：不追加互动提示
        empty_ctx = PromptManager.build_dynamic_context(
            **base, interaction_hint_text="   "
        )
        self.assertNotIn("【好感度系统】", empty_ctx)


class RelationSystemTests(unittest.TestCase):
    def test_rel_tag_parsing_and_cleaning(self):
        for tag, direction in (
            ("[REL:up]", "up"),
            ("[REL:down]", "down"),
            ("[rel:UP]", "up"),
            ("[REL:升]", "up"),
            ("[REL:降]", "down"),
            ("**[REL:up]**", "up"),
            ("【REL:up】", "up"),
        ):
            m = RE_REL.search(f"回复\n{tag}")
            self.assertIsNotNone(m)
            self.assertEqual(normalize_rel_direction(m.group(1)), direction)
            self.assertEqual(clean_tags_from_text(f"回复\n{tag}"), "回复")

    def test_bold_and_fullwidth_tags_cleaning(self):
        text = "你好啊！这是**加粗正文**与`代码块`。\n**[FAV:+2]**\n【EVAL:特别有趣】\n**[STK:开心]**\n【MUTE:60】"
        cleaned = clean_tags_from_text(text)
        self.assertEqual(cleaned, "你好啊！这是**加粗正文**与`代码块`。")

    def test_mute_prompt_has_no_score_requirement(self):
        from services.prompt import FAV_MUTE_PROMPT
        self.assertNotIn("-20", FAV_MUTE_PROMPT)
        self.assertNotIn("好感度不高于", FAV_MUTE_PROMPT)

    def test_score_to_attitude_bands(self):
        self.assertEqual(score_to_attitude(80), "热忱亲昵")
        self.assertEqual(score_to_attitude(60), "温和热情")
        self.assertEqual(score_to_attitude(30), "轻快友好")
        self.assertEqual(score_to_attitude(0), "平静礼貌")
        self.assertEqual(score_to_attitude(-30), "略显微慢")
        self.assertEqual(score_to_attitude(-60), "冷淡克制")
        self.assertEqual(score_to_attitude(-90), "冰冷疏离")

    def test_only_current_relation_guideline_injected(self):
        prompt = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=False,
            prompt_preset="default",
            relation="知心挚友",
        )
        self.assertIn("当前关系行为准则（知心挚友）", prompt)
        self.assertIn(RELATION_GUIDELINES["知心挚友"], prompt)
        self.assertIn(FAV_RELATION_PROMPT, prompt)
        # 其他档位准则不应出现
        for name, text in RELATION_GUIDELINES.items():
            if name != "知心挚友":
                self.assertNotIn(text, prompt)
        self.assertNotIn("当前关系行为准则（挚爱恋人）", prompt)

    def test_relation_section_respects_switch_and_preset(self):
        disabled = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=False,
            prompt_preset="default",
            relation="知心挚友",
            relation_enabled=False,
        )
        self.assertNotIn("当前关系行为准则（", disabled)
        self.assertNotIn("[REL:up]", disabled)
        self.assertNotIn("关系慎重", disabled)

        old_preset = PromptManager.build_static_prompt(
            favorability_enabled=True,
            sticker_enabled=False,
            prompt_preset="old",
            relation="知心挚友",
        )
        self.assertNotIn("当前关系行为准则（", old_preset)
        self.assertNotIn("[REL:up]", old_preset)

    def test_dynamic_context_includes_relation_state(self):
        ctx = PromptManager.build_dynamic_context(
            favorability_enabled=True,
            system_time_enabled=False,
            user_info_enabled=False,
            score=55,
            eval_text="聊得来",
            relation="知心挚友",
            pending_rel={
                "direction": "up",
                "from": "知心挚友",
                "to": "挚爱恋人",
                "expires_at": 0,
            },
        )
        self.assertIn("好感度：55（说话态度：温和热情）", ctx)
        self.assertIn("当前关系：知心挚友", ctx)
        self.assertIn("「知心挚友」→「挚爱恋人」", ctx)
        self.assertIn("不要重复发起关系提议", ctx)

    def test_dynamic_context_skips_relation_when_disabled(self):
        ctx = PromptManager.build_dynamic_context(
            favorability_enabled=True,
            system_time_enabled=False,
            user_info_enabled=False,
            score=55,
            eval_text="聊得来",
            relation="知心挚友",
            relation_enabled=False,
        )
        self.assertNotIn("当前关系", ctx)

    def test_relation_guideline_falls_back_for_unknown(self):
        section = build_relation_guideline("不存在的档位")
        self.assertIn("当前关系行为准则（普通朋友）", section)
        self.assertIn(RELATION_GUIDELINES["普通朋友"], section)


class ConfigSchemaTests(unittest.TestCase):
    def test_prompt_config_schema(self):
        schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        for key in (
            "mute_condition",
            "sticker_condition",
            "favorability_prompt_core",
            "favorability_prompt_behavior",
            "favorability_prompt_relation",
            "favorability_prompt_mute",
            "favorability_prompt_security",
            "relation_guideline_lover",
            "relation_guideline_confidant",
            "relation_guideline_friend",
            "relation_guideline_acquaintance",
            "relation_guideline_estranged",
            "relation_guideline_rival",
            "relation_guideline_severed",
        ):
            self.assertEqual(schema[key]["type"], "text")
            self.assertTrue(schema[key]["default"])
            self.assertEqual(schema[key]["condition"], {"prompt_preset": "custom"})

        self.assertEqual(schema["prompt_preset"]["type"], "string")
        self.assertEqual(schema["prompt_preset"]["default"], "default")
        self.assertEqual(schema["prompt_preset"]["options"], ["default", "old", "custom"])
        schema_keys = list(schema)
        self.assertLess(schema_keys.index("mute_condition"), schema_keys.index("sticker_condition"))
        self.assertLess(schema_keys.index("prompt_preset"), schema_keys.index("mute_condition"))

        expected_groups = {
            "feature_settings": [
                "plugin_enabled",
                "favorability_enabled",
                "sticker_enabled",
                "mute_enabled",
                "relation_enabled",
                "interaction_hint_enabled",
            ],
            "prompt_settings": [
                "prompt_preset",
                "mute_condition",
                "sticker_condition",
                "favorability_prompt_core",
                "favorability_prompt_behavior",
                "favorability_prompt_relation",
                "favorability_prompt_mute",
                "favorability_prompt_security",
                "relation_guideline_lover",
                "relation_guideline_confidant",
                "relation_guideline_friend",
                "relation_guideline_acquaintance",
                "relation_guideline_estranged",
                "relation_guideline_rival",
                "relation_guideline_severed",
                "interaction_hint_text",
            ],
            "persona_settings": ["persona_overrides"],
            "context_settings": ["system_time_enabled", "user_info_enabled"],
            "render_settings": ["render_theme"],
        }
        for group, keys in expected_groups.items():
            self.assertEqual(schema[group]["type"], "object")
            self.assertEqual(list(schema[group]["items"]), keys)

        for key in (
            "mute_condition",
            "sticker_condition",
            "favorability_prompt_core",
            "favorability_prompt_behavior",
            "favorability_prompt_relation",
            "favorability_prompt_mute",
            "favorability_prompt_security",
            "relation_guideline_lover",
            "relation_guideline_confidant",
            "relation_guideline_friend",
            "relation_guideline_acquaintance",
            "relation_guideline_estranged",
            "relation_guideline_rival",
            "relation_guideline_severed",
        ):
            self.assertEqual(
                schema["prompt_settings"]["items"][key]["condition"],
                {"prompt_preset": "custom"},
            )
            self.assertTrue(schema["prompt_settings"]["items"][key]["default"])

        self.assertEqual(
            schema["feature_settings"]["items"]["interaction_hint_enabled"]["type"],
            "bool",
        )
        self.assertTrue(
            schema["feature_settings"]["items"]["interaction_hint_enabled"]["default"]
        )
        self.assertEqual(
            schema["prompt_settings"]["items"]["interaction_hint_text"]["type"], "text"
        )
        self.assertTrue(schema["prompt_settings"]["items"]["interaction_hint_text"]["default"])
        # 跨组 condition 在 WebUI 中不生效，因此该文本项不应带 condition
        self.assertNotIn(
            "condition", schema["prompt_settings"]["items"]["interaction_hint_text"]
        )

        self.assertTrue(schema["_config_layout_version"]["invisible"])
        for key in (
            "plugin_enabled",
            "favorability_enabled",
            "sticker_enabled",
            "prompt_preset",
            "mute_enabled",
            "relation_enabled",
            "mute_condition",
            "sticker_condition",
            "favorability_prompt_core",
            "favorability_prompt_behavior",
            "favorability_prompt_relation",
            "favorability_prompt_mute",
            "favorability_prompt_security",
            "relation_guideline_lover",
            "relation_guideline_confidant",
            "relation_guideline_friend",
            "relation_guideline_acquaintance",
            "relation_guideline_estranged",
            "relation_guideline_rival",
            "relation_guideline_severed",
            "system_time_enabled",
            "user_info_enabled",
            "render_theme",
        ):
            self.assertTrue(schema[key]["invisible"])


if __name__ == "__main__":
    unittest.main()
