"""关系系统与好感度数值解耦的数据层测试。

models.manager 仅依赖 astrbot.api.logger，这里用轻量 stub 使其可脱离
AstrBot 环境运行：
  python -m unittest discover -s tests -t . -p "test_relation_flow.py"
"""

import asyncio
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

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

from models.manager import FavorabilityManager  # noqa: E402


class RelationFlowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.mgr = FavorabilityManager(self.data_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def test_defaults_are_decoupled(self):
        info = self.mgr.get_user_info("g", "alice")
        self.assertEqual(info["relation"], FavorabilityManager.DEFAULT_RELATION)
        self.assertIsNone(info["pending_rel"])
        self.assertEqual(info["score"], 0)

    def test_score_changes_never_move_relation(self):
        async def scenario():
            await self.mgr.update_user("g", "alice", change=5, new_eval="不错")
            await self.mgr.set_score("g", "alice", 999)
            info = self.mgr.get_user_info("g", "alice")
            self.assertEqual(info["score"], 999)
            self.assertEqual(info["relation"], "普通朋友")
            await self.mgr.set_score("g", "alice", -999)
            info = self.mgr.get_user_info("g", "alice")
            self.assertEqual(info["relation"], "普通朋友")

        asyncio.run(scenario())

    def test_propose_then_confirm_adjacent_only(self):
        async def scenario():
            result = await self.mgr.propose_relation("g", "alice", "up")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["pending"]["from"], "普通朋友")
            self.assertEqual(result["pending"]["to"], "熟络好友")

            # 重复提议 → already_pending
            again = await self.mgr.propose_relation("g", "alice", "up")
            self.assertEqual(again["status"], "already_pending")

            applied = await self.mgr.confirm_relation("g", "alice")
            self.assertEqual(applied["status"], "applied")
            self.assertEqual(applied["to"], "熟络好友")
            info = self.mgr.get_user_info("g", "alice")
            self.assertEqual(info["relation"], "熟络好友")
            self.assertIsNone(info["pending_rel"])

            # 确认后没有待确认提议
            none = await self.mgr.confirm_relation("g", "alice")
            self.assertEqual(none["status"], "none")

        asyncio.run(scenario())

    def test_propose_step_by_step_to_top(self):
        async def scenario():
            for target in ("熟络好友", "知心挚友", "挚爱恋人"):
                await self.mgr.propose_relation("g", "alice", "up")
                result = await self.mgr.confirm_relation("g", "alice")
                self.assertEqual(result["status"], "applied")
                self.assertEqual(result["to"], target)
            # 已到顶档，再升 → boundary
            top = await self.mgr.propose_relation("g", "alice", "up")
            self.assertEqual(top["status"], "boundary")
            info = self.mgr.get_user_info("g", "alice")
            self.assertEqual(info["relation"], "挚爱恋人")

        asyncio.run(scenario())

    def test_pending_expires(self):
        async def scenario():
            await self.mgr.propose_relation("g", "alice", "down")
            # 手动改为已过期
            data = self.mgr._read()
            data["g"]["alice"]["pending_rel"]["expires_at"] = time.time() - 1
            self.mgr._write(data)
            expired = await self.mgr.confirm_relation("g", "alice")
            self.assertEqual(expired["status"], "expired")
            info = self.mgr.get_user_info("g", "alice")
            self.assertIsNone(info["pending_rel"])
            self.assertEqual(info["relation"], "普通朋友")
            # 过期后可以重新提议
            ok = await self.mgr.propose_relation("g", "alice", "down")
            self.assertEqual(ok["status"], "ok")
            self.assertEqual(ok["pending"]["to"], "生疏之交")

        asyncio.run(scenario())

    def test_stale_proposal_after_admin_set(self):
        async def scenario():
            await self.mgr.propose_relation("g", "alice", "up")
            await self.mgr.set_relation("g", "alice", "知心挚友")
            stale = await self.mgr.confirm_relation("g", "alice")
            self.assertEqual(stale["status"], "none")  # set_relation 已清空提议
            info = self.mgr.get_user_info("g", "alice")
            self.assertEqual(info["relation"], "知心挚友")

        asyncio.run(scenario())

    def test_reject_relation(self):
        async def scenario():
            await self.mgr.propose_relation("g", "alice", "up")
            pending = await self.mgr.reject_relation("g", "alice")
            self.assertIsNotNone(pending)
            self.assertEqual(pending["to"], "熟络好友")
            info = self.mgr.get_user_info("g", "alice")
            self.assertIsNone(info["pending_rel"])
            self.assertEqual(info["relation"], "普通朋友")
            # 再次取消 → None
            again = await self.mgr.reject_relation("g", "alice")
            self.assertIsNone(again)

        asyncio.run(scenario())

    def test_admin_set_relation_validates_level(self):
        async def scenario():
            ok = await self.mgr.set_relation("g", "alice", "不合对头")
            self.assertTrue(ok)
            info = self.mgr.get_user_info("g", "alice")
            self.assertEqual(info["relation"], "不合对头")
            self.assertEqual(info["score"], 0)  # 关系与数值互不影响
            # 兼容旧名称设置
            legacy_ok = await self.mgr.set_relation("g", "alice", "亲密朋友")
            self.assertTrue(legacy_ok)
            self.assertEqual(self.mgr.get_user_info("g", "alice")["relation"], "知心挚友")
            bad = await self.mgr.set_relation("g", "alice", "陌生人")
            self.assertFalse(bad)

        asyncio.run(scenario())

    def test_reset_restores_default_relation(self):
        async def scenario():
            await self.mgr.set_relation("g", "alice", "挚爱恋人")
            await self.mgr.propose_relation("g", "alice", "down")
            await self.mgr.reset_user("g", "alice")
            info = self.mgr.get_user_info("g", "alice")
            self.assertEqual(info["relation"], "普通朋友")
            self.assertIsNone(info["pending_rel"])
            self.assertEqual(info["score"], 0)

        asyncio.run(scenario())

    def test_legacy_data_migrates_relation_from_score(self):
        data = {
            "g": {
                "bob": {"score": 80, "eval": "旧档"},
                "carol": {"score": -60, "eval": "旧档"},
                "dave": {"score": 5, "eval": "旧档", "relation": "亲密朋友"},
            }
        }
        (self.data_dir / "favorability.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
        mgr = FavorabilityManager(self.data_dir)
        self.assertEqual(mgr.get_user_info("g", "bob")["relation"], "挚爱恋人")
        self.assertEqual(mgr.get_user_info("g", "carol")["relation"], "不合对头")
        # 兼容旧 relation 的迁移升级
        self.assertEqual(mgr.get_user_info("g", "dave")["relation"], "知心挚友")

    def test_next_relation_boundaries(self):
        self.assertIsNone(
            FavorabilityManager.next_relation("挚爱恋人", "up")
        )
        self.assertIsNone(
            FavorabilityManager.next_relation("决裂陌路", "down")
        )
        self.assertEqual(
            FavorabilityManager.next_relation("普通朋友", "down"), "生疏之交"
        )
        self.assertIsNone(FavorabilityManager.next_relation("普通朋友", "sideways"))

    def test_low_score_cannot_propose_up(self):
        async def scenario():
            await self.mgr.set_score("g", "alice", -10)
            res = await self.mgr.propose_relation("g", "alice", "up")
            self.assertEqual(res["status"], "low_score_rejected")
            self.assertEqual(res["current_score"], -10)
            # 但负分可以提议降档
            res_down = await self.mgr.propose_relation("g", "alice", "down")
            self.assertEqual(res_down["status"], "ok")

        asyncio.run(scenario())

    def test_reject_triggers_cooldown(self):
        async def scenario():
            await self.mgr.propose_relation("g", "alice", "up")
            rejected = await self.mgr.reject_relation("g", "alice")
            self.assertIsNotNone(rejected)

            # 拒绝后立即提议 → cooldown
            again = await self.mgr.propose_relation("g", "alice", "up")
            self.assertEqual(again["status"], "cooldown")
            self.assertGreater(again["remaining"], 0)

            # 模拟冷却时间已过
            data = self.mgr._read()
            data["g"]["alice"]["rel_cooldown_until"] = time.time() - 1
            self.mgr._write(data)

            # 冷却过后可再次提议
            ok = await self.mgr.propose_relation("g", "alice", "up")
            self.assertEqual(ok["status"], "ok")

        asyncio.run(scenario())

    def test_atomic_write_integrity(self):
        async def scenario():
            await self.mgr.update_user("g", "alice", change=3, new_eval="测试原子写入")
            self.assertTrue(self.mgr.data_file.exists())
            # 确认没有未清理的临时文件
            tmp_files = list(self.mgr.data_file.parent.glob("fav_tmp_*.json"))
            self.assertEqual(len(tmp_files), 0)
            info = self.mgr.get_user_info("g", "alice")
            self.assertEqual(info["score"], 3)
            self.assertEqual(info["eval"], "测试原子写入")

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main(verbosity=2)
