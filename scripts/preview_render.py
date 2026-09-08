"""生成好感度卡片的亮色 / 暗色预览图。"""

from __future__ import annotations

import sys
import types
from pathlib import Path


class _Logger:
    def warning(self, message):
        print(message)


if "astrbot.api" not in sys.modules:
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    api.logger = _Logger()
    astrbot.api = api
    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from astrbot_plugin_favorability.render.image import FavorabilityRenderer


SAMPLE_RANKING = [
    ("10001", {"name": "星河旅人", "score": 9048, "relation": "亲密无间", "eval": "默契又温柔"}),
    ("10002", {"name": "今天也要开心", "score": 8792, "relation": "亲密朋友", "eval": "值得信赖的朋友"}),
    ("10003", {"name": "柚子汽水", "score": 5608, "relation": "聊得来的熟人", "eval": "相处轻松自然"}),
    ("10004", {"name": "白昼梦游", "score": 3025, "relation": "普通关系", "eval": "还在慢慢了解"}),
    ("10005", {"name": "晚风来信", "score": 3001, "relation": "心存芥蒂", "eval": "礼貌而有距离"}),
    ("10006", {"name": "匿名用户", "score": -1234567, "relation": "关系破裂", "eval": "超长分数压力测试"}),
]


def main() -> None:
    output = ROOT / "docs" / "previews"
    output.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        renderer = FavorabilityRenderer(output, theme=theme)
        card = Path(renderer.render_favorability_card(
            user_name="星河旅人",
            user_id="10001",
            score=1234567,
            evaluation="她觉得你真诚、温柔，也很珍惜每一次认真回应。",
            relation="亲密无间",
        ))
        ranking = Path(renderer.render_ranking_image(SAMPLE_RANKING, group_name="莉卡的茶话会"))
        card.replace(output / f"favorability-card-{theme}.png")
        ranking.replace(output / f"favorability-ranking-{theme}.png")
    print(f"预览图已生成：{output}")


if __name__ == "__main__":
    main()
