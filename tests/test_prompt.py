import json
import unittest
from datetime import datetime
from pathlib import Path

from services.prompt import (
    DEFAULT_STICKER_CONDITION,
    FAV_BEHAVIOR_PROMPT,
    FAV_CORE_PROMPT,
    FAV_SECURITY_PROMPT,
    PromptManager,
    format_system_time,
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
        )
        self.assertNotIn("好感度行为规则", sticker_only)
        self.assertNotIn("[FAV:±N]", sticker_only)
        self.assertIn("[STK:分类名]", sticker_only)

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

    def test_eval_text_limit_is_enforced(self):
        self.assertTrue(validate_eval_text("刚刚聊得来"))
        self.assertTrue(validate_eval_text("一" * 20))
        self.assertFalse(validate_eval_text("一" * 21))

    def test_system_time_includes_chinese_weekday(self):
        self.assertEqual(
            format_system_time(datetime(2026, 8, 9, 14, 30, 0)),
            "2026-08-09 14:30:00 星期日",
        )


class ConfigSchemaTests(unittest.TestCase):
    def test_prompt_config_schema(self):
        schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        for key in (
            "sticker_condition",
            "favorability_prompt_core",
            "favorability_prompt_behavior",
            "favorability_prompt_mute",
            "favorability_prompt_security",
        ):
            self.assertEqual(schema[key]["type"], "text")
            self.assertTrue(schema[key]["default"])


if __name__ == "__main__":
    unittest.main()
