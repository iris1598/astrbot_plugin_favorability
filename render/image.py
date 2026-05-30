"""
PIL 图片渲染模块 - FavorabilityRenderer

渲染好感度查询卡片和好感度排行榜图片。
使用 Pillow 库绘制，自动适配中文字体。
"""

import math
import uuid
from pathlib import Path
from typing import Optional

from astrbot.api import logger

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning(
        "[favorability] Pillow 未安装，图片渲染功能不可用。请执行: pip install Pillow"
    )


# ── 中文字体查找 ──────────────────────────────────────────

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
    "C:/Windows/Fonts/msyhbd.ttc",  # Microsoft YaHei Bold
    "C:/Windows/Fonts/simhei.ttf",  # SimHei
    "C:/Windows/Fonts/simsun.ttc",  # SimSun
    "C:/Windows/Fonts/deng.ttf",  # DengXian
    "C:/Windows/Fonts/yahei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
]


def _load_font(size: int, bold: bool = False) -> "ImageFont.FreeTypeFont":
    """加载中文字体，失败时回退到默认字体。"""
    candidates = _FONT_CANDIDATES
    if bold and len(candidates) > 1:
        # 优先使用粗体变体
        candidates = [
            c for c in candidates if "bd" in c.lower() or "bold" in c.lower()
        ] + candidates
    for path in candidates:
        p = Path(path)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


# ── 等级与颜色映射 ────────────────────────────────────────


def get_level_info(score: int) -> dict:
    """根据好感度分数返回等级信息。

    Returns:
        dict: {title, color_hex, bg_color_hex, description}
    """
    if score >= 70:
        return {
            "title": "挚爱",
            "color": "#FF4757",
            "bg": "#FFE8EA",
            "description": "爱人级，关系已不分彼此",
        }
    elif score >= 50:
        return {
            "title": "挚友",
            "color": "#FF6348",
            "bg": "#FFE4DB",
            "description": "挚友/恋人级，热情主动",
        }
    elif score >= 21:
        return {
            "title": "熟人",
            "color": "#2ED573",
            "bg": "#E4FCEF",
            "description": "熟人级，积极友好",
        }
    elif score >= -20:
        return {
            "title": "路人",
            "color": "#747D8C",
            "bg": "#F0F0F1",
            "description": "陌生人级，礼貌中性",
        }
    elif score >= -50:
        return {
            "title": "生厌",
            "color": "#FFA502",
            "bg": "#FFF3E0",
            "description": "反感和警惕，对其行为表示厌恶",
        }
    elif score >= -70:
        return {
            "title": "憎恶",
            "color": "#FF4757",
            "bg": "#FFEBEE",
            "description": "极度厌恶，对其行为进行指责谩骂",
        }
    else:
        return {
            "title": "仇敌",
            "color": "#2F3542",
            "bg": "#E8E8E8",
            "description": "光是看到就令人发狂",
        }


def _hex_to_rgb(hex_color: str) -> tuple:
    """将 #RRGGBB 或 #RGB 格式转为 (R, G, B) 元组。"""
    hex_color = hex_color.lstrip("#")
    # 处理简写 #RGB → #RRGGBB
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


# ── 渲染器 ────────────────────────────────────────────────


class FavorabilityRenderer:
    """好感度图片渲染器。"""

    def __init__(self, render_dir: str | Path):
        if not HAS_PIL:
            raise RuntimeError("Pillow 未安装，无法使用图片渲染。")
        self.render_dir = Path(render_dir)
        self.render_dir.mkdir(parents=True, exist_ok=True)

    def _save_img(self, img: "Image.Image") -> str:
        """保存 PIL Image 到文件，返回文件路径。"""
        filename = f"fav_{uuid.uuid4().hex[:12]}.png"
        out_path = str(self.render_dir / filename)
        img.save(out_path, format="PNG")
        return out_path

    # ── 好感度查询卡片 ─────────────────────────────────────

    def render_favorability_card(
        self,
        user_name: str,
        user_id: str,
        score: int,
        evaluation: str,
    ) -> str:
        """渲染一张用户好感度查询卡片，返回 PNG 文件路径。"""
        W, H = 480, 280
        img = Image.new("RGB", (W, H), "#FFFFFF")
        draw = ImageDraw.Draw(img)

        level = get_level_info(score)
        accent = _hex_to_rgb(level["color"])
        accent_orange = _hex_to_rgb(self.ORANGE_ACCENT)

        # 字体
        font_lg = _load_font(36, bold=True)
        font_md = _load_font(20)
        font_sm = _load_font(16)
        font_title = _load_font(26, bold=True)

        # ── 顶部色带 ──
        draw.rectangle([(0, 0), (W, 8)], fill=accent)

        # ── 等级徽章（左上角） ──
        badge_w, badge_h = 80, 32
        badge_x, badge_y = 20, 24
        draw.rounded_rectangle(
            [(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)],
            radius=16,
            fill=accent,
        )
        bbox = draw.textbbox((0, 0), level["title"], font=font_sm)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (badge_x + (badge_w - tw) // 2, badge_y + (badge_h - th) // 2 - 1),
            level["title"],
            fill="#FFFFFF",
            font=font_sm,
        )

        # ── 用户名称 ──
        draw.text(
            (20, 68), user_name, fill=_hex_to_rgb(self.TEXT_DARK), font=font_title
        )

        # ── 用户 ID ──
        draw.text(
            (20, 100), f"ID: {user_id}", fill=_hex_to_rgb(self.TEXT_MUTED), font=font_sm
        )

        # ── 好感度分数（右侧大号数字） ──
        score_text = f"{score:+d}" if score != 0 else "0"
        bbox = draw.textbbox((0, 0), score_text, font=font_lg)
        sw = bbox[2] - bbox[0]
        draw.text(
            (W - sw - 30, 28),
            score_text,
            fill=accent,
            font=font_lg,
        )
        draw.text(
            (W - 120, 72), "好 感 度", fill=_hex_to_rgb(self.TEXT_MUTED), font=font_sm
        )

        # ── 分隔线 ──
        draw.line([(20, 140), (W - 20, 140)], fill="#E8E8E8", width=1)

        # ── 评价 ──
        draw.text(
            (20, 156), "她的评价", fill=_hex_to_rgb(self.TEXT_MUTED), font=font_sm
        )
        eval_text = evaluation if evaluation else "暂无"
        draw.text((20, 184), eval_text, fill=_hex_to_rgb(self.TEXT_DARK), font=font_md)

        # ── 等级描述 ──
        draw.text(
            (20, 224),
            level["description"],
            fill=_hex_to_rgb(self.TEXT_MUTED),
            font=font_sm,
        )

        # ── 底部装饰条 ──
        draw.rectangle([(0, H - 4), (W, H)], fill=accent)

        return self._save_img(img)

    # ── 好感度排行（橙色主题） ─────────────────────────────

    ORANGE_LIGHT = "#FFF5EB"
    ORANGE_ACCENT = "#FF8C00"
    ORANGE_DARK = "#E67E00"
    ORANGE_MUTED = "#FFB347"
    ORANGE_BG = "#FFFAF5"
    TEXT_DARK = "#2C3E50"
    TEXT_MUTED = "#95A5A6"
    TEXT_WHITE = "#FFFFFF"

    def render_ranking_image(
        self,
        ranked_list: list[tuple[str, dict]],
        group_name: str = "",
        ascending: bool = False,
    ) -> str:
        """渲染好感度排行榜图片（橙色主题），返回 PNG 文件路径。

        Args:
            ascending: True 时标题显示"好感度低谷榜"（正序最低分在前）
        """
        n = len(ranked_list)
        if n == 0:
            return self.render_empty_ranking()

        row_h = 46
        header_h = 54
        footer_h = 40
        pad = 24
        W = 520
        title_h = 64

        list_h = n * row_h
        total_h = title_h + header_h + list_h + footer_h + pad * 2

        img = Image.new("RGB", (W, total_h), self.ORANGE_BG)
        draw = ImageDraw.Draw(img)

        font_title = _load_font(26, bold=True)
        font_sub = _load_font(14)
        font_header = _load_font(15, bold=True)
        font_row = _load_font(15)
        font_badge = _load_font(14, bold=True)
        font_footer = _load_font(14)

        y = pad

        # ── 顶栏装饰条 ──
        draw.rectangle([(0, 0), (W, 6)], fill=_hex_to_rgb(self.ORANGE_ACCENT))

        # ── 标题区 ──
        # 标题背景
        draw.rounded_rectangle(
            [(pad, y), (W - pad, y + title_h - 4)],
            radius=12,
            fill=_hex_to_rgb(self.ORANGE_LIGHT),
        )
        # 标题文字
        title_text = "好感度倒序榜" if ascending else "好感度排行榜"
        bbox = draw.textbbox((0, 0), title_text, font=font_title)
        tw = bbox[2] - bbox[0]
        draw.text(
            ((W - tw) // 2, y + 12),
            title_text,
            fill=_hex_to_rgb(self.ORANGE_ACCENT),
            font=font_title,
        )
        # 副标题
        sub_prefix = "倒数" if ascending else "前"
        sub_text = f"共 {n} 位用户 · {sub_prefix}{n} 名基于好感度总分"
        bbox = draw.textbbox((0, 0), sub_text, font=font_sub)
        sw = bbox[2] - bbox[0]
        draw.text(
            ((W - sw) // 2, y + 42),
            sub_text,
            fill=_hex_to_rgb(self.TEXT_MUTED),
            font=font_sub,
        )
        y += title_h

        # ── 表头 ──
        draw.rounded_rectangle(
            [(pad, y), (W - pad, y + header_h)],
            radius=10,
            fill=_hex_to_rgb(self.ORANGE_ACCENT),
        )
        # 表头列宽：排名 50px | 昵称 170px | 分数 100px | 评价 剩余
        col_x = [pad + 16, pad + 60, pad + 228, pad + 326]
        headers = ["#", "昵称", "好感度", "评价"]
        for cx, hdr in zip(col_x, headers):
            draw.text(
                (cx, y + (header_h - 20) // 2),
                hdr,
                fill=self.TEXT_WHITE,
                font=font_header,
            )
        y += header_h

        # ── 数据行 ──
        # 前三名特殊颜色
        top3_bg = ["#FFF3E0", "#FFF8E7", "#FFFDE7"]
        top3_rank_text = ["No.1", "No.2", "No.3"]
        top3_rank_color = ["#FF6B00", "#E8920A", "#B8860B"]
        top3_stripe_color = [
            _hex_to_rgb("#FFE0B2"),
            _hex_to_rgb("#FFECB3"),
            _hex_to_rgb("#FFF9C4"),
        ]

        for i, (uid, udata) in enumerate(ranked_list):
            score = udata.get("score", 0)
            eval_text = udata.get("eval", "")
            display_eval = (eval_text[:10] + "…") if len(eval_text) > 10 else eval_text
            user_name = udata.get("name", "") or uid
            display_name = user_name if len(user_name) <= 10 else user_name[:10] + "…"

            # 行背景
            if i < 3:
                row_bg = top3_bg[i]
            else:
                row_bg = "#FFFFFF" if i % 2 == 0 else "#FFF8F0"
            draw.rounded_rectangle(
                [(pad, y), (W - pad, y + row_h)],
                radius=6,
                fill=row_bg,
            )

            # 排名指示器
            if i < 3:
                # 前三名：带编号徽章和左侧彩色条纹
                draw.rectangle(
                    [(pad, y), (pad + 4, y + row_h)],
                    fill=top3_stripe_color[i],
                )
                # 排名徽章
                badge_w, badge_h = 44, 24
                badge_x = col_x[0] - 4
                badge_y_off = y + (row_h - badge_h) // 2
                draw.rounded_rectangle(
                    [
                        (badge_x, badge_y_off),
                        (badge_x + badge_w, badge_y_off + badge_h),
                    ],
                    radius=12,
                    fill=_hex_to_rgb(top3_rank_color[i]),
                )
                bbox = draw.textbbox((0, 0), top3_rank_text[i], font=font_badge)
                bw = bbox[2] - bbox[0]
                draw.text(
                    (badge_x + (badge_w - bw) // 2, badge_y_off + 3),
                    top3_rank_text[i],
                    fill=self.TEXT_WHITE,
                    font=font_badge,
                )
            else:
                # 普通排名：灰色圆点
                dot_cy = y + row_h // 2
                draw.ellipse(
                    [(col_x[0] + 8, dot_cy - 6), (col_x[0] + 20, dot_cy + 6)],
                    fill=_hex_to_rgb("#DDDDDD"),
                )
                draw.text(
                    (col_x[0] + 8, y + 13),
                    f"{i + 1}",
                    fill=_hex_to_rgb(self.TEXT_MUTED),
                    font=font_row,
                )

            # 用户名
            name_color = _hex_to_rgb(self.TEXT_DARK)
            draw.text((col_x[1], y + 13), display_name, fill=name_color, font=font_row)

            # 好感度分数（按等级着色）
            level = get_level_info(score)
            score_color = _hex_to_rgb(level["color"])
            score_text = f"{score:+d}" if score != 0 else "0"
            # 分数用粗体
            font_score = _load_font(16, bold=True)
            draw.text((col_x[2], y + 12), score_text, fill=score_color, font=font_score)

            # 评价
            draw.text(
                (col_x[3], y + 13),
                display_eval,
                fill=_hex_to_rgb("#57606F"),
                font=font_row,
            )

            y += row_h

        # ── 底部 ──
        y += 8
        draw.rounded_rectangle(
            [(pad, y), (W - pad, y + footer_h - 8)],
            radius=10,
            fill=_hex_to_rgb(self.ORANGE_LIGHT),
        )
        draw.text(
            (W // 2 - 120, y + 10),
            "发送「查询好感度」查看你的详细档案",
            fill=_hex_to_rgb(self.ORANGE_MUTED),
            font=font_footer,
        )

        # ── 底栏装饰条 ──
        draw.rectangle(
            [(0, total_h - 6), (W, total_h)], fill=_hex_to_rgb(self.ORANGE_ACCENT)
        )

        return self._save_img(img)

    # ── 空排行占位 ─────────────────────────────────────────

    def render_empty_ranking(self) -> str:
        W, H = 400, 200
        img = Image.new("RGB", (W, H), self.ORANGE_BG)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (W, 4)], fill=_hex_to_rgb(self.ORANGE_ACCENT))
        font = _load_font(20)
        draw.text(
            (70, 70), "暂无好感度记录", fill=_hex_to_rgb(self.ORANGE_MUTED), font=font
        )
        font_sm = _load_font(14)
        draw.text(
            (90, 110),
            "让 AI 聊起来后自动生成",
            fill=_hex_to_rgb(self.TEXT_MUTED),
            font=font_sm,
        )
        draw.rectangle([(0, H - 4), (W, H)], fill=_hex_to_rgb(self.ORANGE_ACCENT))
        return self._save_img(img)
