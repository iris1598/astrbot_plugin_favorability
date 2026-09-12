import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

for _cand in [
    os.environ.get("ASTRBOT_APP", ""),
    str(PLUGIN_ROOT.parent / "AstrBot" / "backend" / "app"),
]:
    if _cand and (Path(_cand) / "astrbot").is_dir() and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from models.manager import FavorabilityManager, extract_user_id
except ImportError:
    HAS_ASTRBOT = False
    FavorabilityManager = None
    extract_user_id = None
else:
    HAS_ASTRBOT = True


@unittest.skipUnless(HAS_ASTRBOT, "manager 需要 astrbot 环境")
class TestOpenIDPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="fav_test_openid_"))
        self.mgr = FavorabilityManager(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_extract_user_id(self):
        """测试各种格式下 extract_user_id 不会截断十六进制 OpenID。"""
        # 32 位 Hex OpenID 必须原样保留
        openid = "2EBE845D4DFFAB7CD67DAF9A92880A46"
        self.assertEqual(extract_user_id(openid), openid)

        openid_with_leading_digits = "039A4907D3AFDBBB3A3DE4417FC23B72"
        self.assertEqual(extract_user_id(openid_with_leading_digits), openid_with_leading_digits)

        # 带 @ 前缀
        self.assertEqual(extract_user_id(f"@{openid}"), openid)
        self.assertEqual(extract_user_id("@123456789"), "123456789")

        # QQ @ 提及（带括号）
        self.assertEqual(extract_user_id(f"@希({openid})"), openid)
        self.assertEqual(extract_user_id("@萤火虫(039A4907D3AFDBBB3A3DE4417FC23B72)"), "039A4907D3AFDBBB3A3DE4417FC23B72")
        self.assertEqual(extract_user_id("@测试用户(123456789)"), "123456789")

        # 纯数字 QQ
        self.assertEqual(extract_user_id("123456789"), "123456789")

    def test_migrate_heals_corrupted_json(self):
        """测试启动时自愈合并被历史版本截断的 OpenID 档案。"""
        corrupted_data = {
            "爱弥斯:GroupMessage:047CBDC5A372A003834919C42D8EDFA6": {
                "2": {
                    "score": 4,
                    "relation": "普通朋友",
                    "pending_rel": None,
                    "eval": "吐槽我想开高达去堵人的漂泊者",
                    "name": "希",
                    "muted_until": None,
                    "rel_cooldown_until": None,
                },
                "039": {
                    "score": 15,
                    "relation": "普通朋友",
                    "pending_rel": None,
                    "eval": "询问我虚质空间能力的漂泊者",
                    "name": "萤火虫",
                    "muted_until": None,
                    "rel_cooldown_until": None,
                },
                "062": {
                    "score": 2,
                    "relation": "普通朋友",
                    "pending_rel": None,
                    "eval": "配合开玩笑宣誓效忠的漂泊者",
                    "name": "〇ω〇",
                    "muted_until": None,
                    "rel_cooldown_until": None,
                },
                "039A4907D3AFDBBB3A3DE4417FC23B72": {
                    "score": 8,
                    "relation": "普通朋友",
                    "pending_rel": None,
                    "eval": "替西格莉卡传达心事的漂泊者",
                    "name": "萤火虫",
                    "muted_until": None,
                    "rel_cooldown_until": None,
                },
                "2EBE845D4DFFAB7CD67DAF9A92880A46": {
                    "score": 2,
                    "relation": "普通朋友",
                    "pending_rel": None,
                    "eval": "投喂我烤蕨团的漂泊者",
                    "name": "希",
                    "muted_until": None,
                    "rel_cooldown_until": None,
                },
            },
            "爱弥斯:FriendMessage:D2F876A6741C7E3EB8A899BC726EDE7B": {
                "2": {
                    "score": 2,
                    "relation": "普通朋友",
                    "pending_rel": None,
                    "eval": "初识的礼貌来客",
                    "name": "",
                    "muted_until": None,
                    "rel_cooldown_until": None,
                }
            },
        }

        # 写入损坏数据
        data_file = self.test_dir / "favorability.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(corrupted_data, f, ensure_ascii=False)

        # 重新初始化 manager 触发启动迁移与自愈
        mgr2 = FavorabilityManager(self.test_dir)

        # 检查群聊数据自愈结果
        group_key = "爱弥斯:GroupMessage:047CBDC5A372A003834919C42D8EDFA6"
        data = mgr2._read("default")
        users = data[group_key]

        # 短 key "2" 与 "039" 必须已被吸收删除
        self.assertNotIn("2", users)
        self.assertNotIn("039", users)

        # 希 的好感度应累加: 4 + 2 = 6，保留最新评价
        xi = users["2EBE845D4DFFAB7CD67DAF9A92880A46"]
        self.assertEqual(xi["score"], 6)
        self.assertEqual(xi["name"], "希")
        self.assertEqual(xi["eval"], "投喂我烤蕨团的漂泊者")

        # 萤火虫 的好感度应累加: 15 + 8 = 23
        yhc = users["039A4907D3AFDBBB3A3DE4417FC23B72"]
        self.assertEqual(yhc["score"], 23)
        self.assertEqual(yhc["name"], "萤火虫")
        self.assertEqual(yhc["eval"], "替西格莉卡传达心事的漂泊者")

        # 〇ω〇 的孤立短 key "062" 暂时保留等待该用户发话自愈
        self.assertIn("062", users)
        self.assertEqual(users["062"]["score"], 2)

        # 私聊 FriendMessage 中的 "2" 应恢复为真实 friend_id "D2F876A6741C7E3EB8A899BC726EDE7B"
        friend_key = "爱弥斯:FriendMessage:D2F876A6741C7E3EB8A899BC726EDE7B"
        f_users = data[friend_key]
        self.assertNotIn("2", f_users)
        self.assertIn("D2F876A6741C7E3EB8A899BC726EDE7B", f_users)
        self.assertEqual(f_users["D2F876A6741C7E3EB8A899BC726EDE7B"]["score"], 2)

    def test_repeated_restarts_do_not_truncate_or_reset(self):
        """测试多次重启不会发生截断或重置好感度。"""
        group_key = "test:GroupMessage:GROUP123"
        openid = "2EBE845D4DFFAB7CD67DAF9A92880A46"

        # 用户交互，获得好感度
        asyncio.run(
            self.mgr.update_user(group_key, openid, change=3, new_eval="测试评价", user_name="希")
        )

        info1 = self.mgr.get_user_info(group_key, openid)
        self.assertEqual(info1["score"], 3)
        self.assertEqual(info1["eval"], "测试评价")

        # 模拟多次重启
        for _ in range(3):
            mgr_restart = FavorabilityManager(self.test_dir)
            info_after = mgr_restart.get_user_info(group_key, openid)
            self.assertEqual(info_after["score"], 3)
            self.assertEqual(info_after["eval"], "测试评价")
            # 确认数据库里 key 依然是 32 位完整 OpenID
            raw = mgr_restart._read("default")
            self.assertIn(openid, raw[group_key])
            self.assertNotIn("2", raw[group_key])

    def test_dynamic_healing_for_orphan_short_key(self):
        """测试孤立短 key 在用户首次发言（带完整 OpenID）时自动继承升级。"""
        group_key = "test:GroupMessage:GROUP123"
        legacy_data = {
            group_key: {
                "062": {
                    "score": 5,
                    "relation": "熟络好友",
                    "pending_rel": None,
                    "eval": "配合开玩笑宣誓效忠的漂泊者",
                    "name": "〇ω〇",
                    "muted_until": None,
                    "rel_cooldown_until": None,
                }
            }
        }
        data_file = self.test_dir / "favorability.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f, ensure_ascii=False)

        mgr = FavorabilityManager(self.test_dir)
        full_openid = "062F8BACD1234567890ABCDEF1234567"

        # 1. 构造 prompt 阶段查好感度，应能无缝返回孤立短 key 的好感档案（不显示初次见面）
        info = mgr.get_user_info(group_key, full_openid)
        self.assertEqual(info["score"], 5)
        self.assertEqual(info["relation"], "熟络好友")
        self.assertEqual(info["eval"], "配合开玩笑宣誓效忠的漂泊者")

        # 2. LLM 响应后 update_user，应无缝迁移并升级为 full_openid
        updated = asyncio.run(
            mgr.update_user(group_key, full_openid, change=2, new_eval="新互动评价", user_name="〇ω〇")
        )
        self.assertEqual(updated["score"], 7)

        # 3. 检查落盘文件：短 key "062" 已被消除，完全升级为 full_openid
        saved = mgr._read("default")
        self.assertNotIn("062", saved[group_key])
        self.assertIn(full_openid, saved[group_key])
        self.assertEqual(saved[group_key][full_openid]["score"], 7)
        self.assertEqual(saved[group_key][full_openid]["eval"], "新互动评价")


if __name__ == "__main__":
    unittest.main()
