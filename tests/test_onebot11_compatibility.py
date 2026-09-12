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
class TestOneBot11Compatibility(unittest.TestCase):
    def setUp(self):
        self.sample_file = Path(r"C:\Users\84637\Desktop\code\favorabilityonebot11.json")
        self.test_dir = Path(tempfile.mkdtemp(prefix="fav_test_onebot11_"))
        self.test_json = self.test_dir / "favorability.json"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_onebot11_data_integrity(self):
        """测试加载真实 OneBot11 数据时，所有 QQ 号、分数、关系、评价完全无损。"""
        if not self.sample_file.exists():
            self.skipTest("favorabilityonebot11.json 不存在")

        with open(self.sample_file, "r", encoding="utf-8") as f:
            orig_data = json.load(f)

        shutil.copy(self.sample_file, self.test_json)

        # 首次启动加载并执行迁移
        mgr = FavorabilityManager(self.test_dir)
        migrated_data = mgr._read("default")

        # 筛选 OneBot11 群聊和私聊会话
        onebot_groups = [
            k for k in orig_data.keys()
            if any(id_str in k for id_str in ["1041386550", "905742726", "559026815", "FriendMessage:248690058"])
        ]

        total_checked = 0
        for gk in onebot_groups:
            orig_users = orig_data[gk]
            mig_users = migrated_data.get(gk, {})

            for uid, orig_entry in orig_users.items():
                total_checked += 1
                # 1. 用户 ID 必须依然是原本的纯数字 QQ
                self.assertIn(uid, mig_users, f"OneBot11 用户 {uid} 在群 {gk} 中丢失！")
                mig_entry = mig_users[uid]

                # 2. 分数必须完全相同
                self.assertEqual(
                    mig_entry.get("score"),
                    orig_entry.get("score"),
                    f"用户 {uid} 分数不一致！",
                )

                # 3. 昵称必须完全相同
                self.assertEqual(
                    mig_entry.get("name"),
                    orig_entry.get("name"),
                    f"用户 {uid} 昵称不一致！",
                )

                # 4. 评价必须完全相同
                self.assertEqual(
                    mig_entry.get("eval"),
                    orig_entry.get("eval"),
                    f"用户 {uid} 评价不一致！",
                )

                # 5. 关系档位必须完全兼容保留
                self.assertEqual(
                    mig_entry.get("relation"),
                    orig_entry.get("relation"),
                    f"用户 {uid} 关系档位不一致！",
                )

        print(f"\n[TestOneBot11] 成功校验 {total_checked} 位 OneBot11 用户数据，0 丢失、0 偏差！")

        # 验证 OneBot11 高频用户（如 846370266，好感度 3755）
        user_info = mgr.get_user_info("Iris:GroupMessage:1041386550", "846370266")
        self.assertEqual(user_info["score"], 3755)
        self.assertEqual(user_info["relation"], "挚爱恋人")
        self.assertEqual(user_info["name"], "xp是摆烂")

        # 模拟连续重启 3 次，确认 OneBot11 数据绝不发生漂移或截断
        for _ in range(3):
            mgr_restart = FavorabilityManager(self.test_dir)
            info_after = mgr_restart.get_user_info("Iris:GroupMessage:1041386550", "846370266")
            self.assertEqual(info_after["score"], 3755)
            self.assertEqual(info_after["relation"], "挚爱恋人")

    def test_onebot11_at_and_user_commands(self):
        """测试 OneBot11 常规的 @ 格式与指令解析是否受影响。"""
        # 纯数字 QQ 提取
        self.assertEqual(extract_user_id("846370266"), "846370266")
        # 带 @ 纯数字
        self.assertEqual(extract_user_id("@846370266"), "846370266")
        # OneBot QQ 提及标准格式: @昵称(846370266)
        self.assertEqual(extract_user_id("@xp是摆烂(846370266)"), "846370266")
        self.assertEqual(extract_user_id("@永远喜欢西格莉卡(248690058)"), "248690058")

    def test_onebot11_and_qqofficial_coexistence(self):
        """测试 OneBot11 纯数字 QQ 与 qqofficial_full 32位 Hex OpenID 在同一数据库中共存。"""
        mgr = FavorabilityManager(self.test_dir)

        # 1. 写入 OneBot11 群与用户
        ob_gk = "Iris:GroupMessage:1041386550"
        ob_uid = "846370266"
        asyncio.run(mgr.update_user(ob_gk, ob_uid, change=5, new_eval="OneBot用户交互", user_name="OB用户"))

        # 2. 写入 qqofficial_full 群与用户
        qq_gk = "Iris:GroupMessage:047CBDC5A372A003834919C42D8EDFA6"
        qq_uid = "2EBE845D4DFFAB7CD67DAF9A92880A46"
        asyncio.run(mgr.update_user(qq_gk, qq_uid, change=5, new_eval="官方机器人用户交互", user_name="希"))

        # 3. 重启插件
        mgr2 = FavorabilityManager(self.test_dir)

        # 4. 验证互不影响
        ob_res = mgr2.get_user_info(ob_gk, ob_uid)
        self.assertEqual(ob_res["score"], 5)
        self.assertEqual(ob_res["name"], "OB用户")

        qq_res = mgr2.get_user_info(qq_gk, qq_uid)
        self.assertEqual(qq_res["score"], 5)
        self.assertEqual(qq_res["name"], "希")


if __name__ == "__main__":
    unittest.main()
