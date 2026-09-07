"""group_storage_key 单元测试（插件修复核心）。

用 AstrBot 自带 Python 运行（manager.py 需要 astrbot 可导入）：
  AstrBot\\backend\\python\\python.exe -m pytest astrbot_plugin_favorability/tests -v
或直接：
  AstrBot\\backend\\python\\python.exe astrbot_plugin_favorability\\tests\\test_group_storage_key.py
"""

import os
import sys
import unittest
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

# AstrBot 便携版的 app 目录（含 astrbot 包）：相对仓库根或 ASTRBOT_APP 环境变量
for _cand in [os.environ.get("ASTRBOT_APP", ""),
              str(PLUGIN_ROOT.parent / "AstrBot" / "backend" / "app")]:
    if _cand and (Path(_cand) / "astrbot").is_dir() and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from models.manager import group_storage_key
except ImportError:
    HAS_ASTRBOT = False
    group_storage_key = None
else:
    HAS_ASTRBOT = True


@unittest.skipUnless(HAS_ASTRBOT, "manager 需要 astrbot 环境")
class TestGroupStorageKey(unittest.TestCase):
    def test_strip_isolated_prefix(self):
        self.assertEqual(
            group_storage_key("Iris:GroupMessage:846370266_1041386550", "846370266"),
            "Iris:GroupMessage:1041386550",
        )

    def test_legacy_group_shared_unchanged(self):
        # 未开启隔离：session_id 就是群号，不以 sender_ 开头
        self.assertEqual(
            group_storage_key("Iris:GroupMessage:905742726", "846370266"),
            "Iris:GroupMessage:905742726",
        )

    def test_group_number_that_could_be_mistaken(self):
        # 群号本身带下划线场景（非 QQ）：前缀不匹配 sender 时不动
        self.assertEqual(
            group_storage_key("Iris:GroupMessage:123456_789", "111111"),
            "Iris:GroupMessage:123456_789",
        )
        # 恰好匹配则剥离（unique_session 标准格式）
        self.assertEqual(
            group_storage_key("Iris:GroupMessage:123456_789", "123456"),
            "Iris:GroupMessage:789",
        )

    def test_private_chat_unchanged(self):
        self.assertEqual(
            group_storage_key("Iris:FriendMessage:846370266", "846370266"),
            "Iris:FriendMessage:846370266",
        )
        self.assertEqual(
            group_storage_key(
                "webchat:FriendMessage:webchat!miku1598!uuid-1", "1598"),
            "webchat:FriendMessage:webchat!miku1598!uuid-1",
        )

    def test_other_platform_unchanged(self):
        # qq_official_full 群 sid 无下划线
        self.assertEqual(
            group_storage_key(
                "default_1903658496:GroupMessage:0A7992D28365E7196D22AC8A64672D90",
                "96976CB5E822D7A13E71CA8D78BF16A0",
            ),
            "default_1903658496:GroupMessage:0A7992D28365E7196D22AC8A64672D90",
        )

    def test_exact_sender_without_group_empty(self):
        # session_id 恰好等于 "{sender}_"（无群号）→ 不剥离
        self.assertEqual(
            group_storage_key("Iris:GroupMessage:123_", "123"),
            "Iris:GroupMessage:123_",
        )

    def test_all_members_converge_same_bucket(self):
        a = group_storage_key("Iris:GroupMessage:111_559026815", "111")
        b = group_storage_key("Iris:GroupMessage:222_559026815", "222")
        self.assertEqual(a, b)
        # 管理员禁言目标 → 目标 enforcement 同桶可见
        admin_bucket = group_storage_key(
            "Iris:GroupMessage:999_559026815", "999")
        target_bucket = group_storage_key(
            "Iris:GroupMessage:111_559026815", "111")
        self.assertEqual(admin_bucket, target_bucket)


if __name__ == "__main__":
    unittest.main(verbosity=2)
