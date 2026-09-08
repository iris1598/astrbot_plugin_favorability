"""好感度图片渲染模块。

视觉风格沿用 astrbot_plugin_rika_share 的现代分享卡片语言：圆角渐变卡片、
柔和品牌色光晕、半透明信息胶囊、细描边与克制阴影，并提供 light / dark
两套主题。全部使用 Pillow 绘制，不依赖浏览器或网络资源。
"""

from __future__ import annotations

import math
import time
import uuid
from pathlib import Path

from astrbot.api import logger

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    HAS_PIL = True
except ImportError:  # pragma: no cover
    HAS_PIL = False
    Image = ImageDraw = ImageFilter = ImageFont = None
    logger.warning("[favorability] Pillow 未安装，图片渲染功能不可用。")


_FONT_CANDIDATES = {
    "regular": [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/deng.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ],
    "bold": [
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/dengb.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ],
}


def _find_font(bold: bool = False) -> str | None:
    key = "bold" if bold else "regular"
    for candidate in _FONT_CANDIDATES[key]:
        if Path(candidate).exists():
            return candidate
    return None


_FONT_PATHS = {"regular": _find_font(), "bold": _find_font(True)}
_FONT_CACHE: dict[tuple[int, bool], "ImageFont.FreeTypeFont"] = {}


def _load_font(size: int, bold: bool = False) -> "ImageFont.FreeTypeFont":
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = _FONT_PATHS["bold" if bold else "regular"] or _FONT_PATHS["regular"]
    try:
        font = ImageFont.truetype(path, size=size) if path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _rgba(value: str | tuple[int, int, int], alpha: int = 255) -> tuple[int, int, int, int]:
    rgb = _hex(value) if isinstance(value, str) else value
    return (*rgb, alpha)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * ratio) for i in range(3))


_RELATION_INFO = {
    "亲密无间": {"color": "#FF5C7A", "description": "重要的人 · 自然亲昵而热情"},
    "亲密朋友": {"color": "#FF7A5C", "description": "亲密关系 · 主动关心与互动"},
    "聊得来的熟人": {"color": "#36C98F", "description": "熟人关系 · 轻松而友好"},
    "普通关系": {"color": "#8290A8", "description": "一般关系 · 礼貌而有分寸"},
    "心存芥蒂": {"color": "#F2A93B", "description": "有所介意 · 谨慎并保持边界"},
    "明显反感": {"color": "#F05B68", "description": "关系紧张 · 冷淡而坚定"},
    "关系破裂": {"color": "#8B5CF6", "description": "接近冰点 · 减少互动并直接拒绝"},
}


def get_relation_info(relation: str | None) -> dict:
    """根据关系档位返回徽章信息；未知档位回退为普通关系。"""
    name = relation if relation in _RELATION_INFO else "普通关系"
    info = _RELATION_INFO[name]
    return {"title": name, "color": info["color"], "description": info["description"]}


def relation_for_score(score: int) -> str:
    """兼容旧数据：无关系字段时按历史分数区间推导。"""
    if score >= 70:
        return "亲密无间"
    if score >= 50:
        return "亲密朋友"
    if score >= 21:
        return "聊得来的熟人"
    if score >= -20:
        return "普通关系"
    if score >= -50:
        return "心存芥蒂"
    if score >= -70:
        return "明显反感"
    return "关系破裂"


def get_level_info(score: int) -> dict:
    """（已弃用）按分数返回等级信息，仅供旧调用兼容。"""
    return get_relation_info(relation_for_score(score))


_THEMES = {
    "dark": {
        "top": "#242B3F",
        "bottom": "#12161F",
        "surface": "#FFFFFF",
        "text": "#F5F7FC",
        "secondary": "#AEB6C8",
        "tertiary": "#78849A",
        "border": "#FFFFFF",
        "shadow": 135,
        "glow": 34,
        "surface_alpha": 14,
        "surface_strong": 22,
        "border_alpha": 25,
    },
    "light": {
        "top": "#FFFFFF",
        "bottom": "#F1F4F9",
        "surface": "#182033",
        "text": "#1A2130",
        "secondary": "#55607A",
        "tertiary": "#8C95A9",
        "border": "#1B2233",
        "shadow": 48,
        "glow": 18,
        "surface_alpha": 9,
        "surface_strong": 14,
        "border_alpha": 16,
    },
}


class FavorabilityRenderer:
    """现代双主题好感度图片渲染器。"""

    def __init__(
        self,
        render_dir: str | Path,
        cache_max_age: int = 3600,
        theme: str = "dark",
    ):
        if not HAS_PIL:
            raise RuntimeError("Pillow 未安装，无法使用图片渲染。")
        self.render_dir = Path(render_dir)
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self.cache_max_age = cache_max_age
        self.theme_name = theme if theme in _THEMES else "dark"

    @property
    def theme(self) -> dict:
        return _THEMES[self.theme_name]

    def cleanup_cache(self, max_age: int | None = None) -> tuple[int, int]:
        max_age = self.cache_max_age if max_age is None else max_age
        now = time.time()
        deleted = remaining = 0
        if not self.render_dir.exists():
            return 0, 0
        for file in self.render_dir.iterdir():
            if not file.is_file() or not file.name.startswith("fav_") or file.suffix != ".png":
                continue
            try:
                if now - file.stat().st_mtime > max_age:
                    file.unlink()
                    deleted += 1
                else:
                    remaining += 1
            except OSError:
                remaining += 1
        return deleted, remaining

    def get_cache_info(self) -> dict:
        now = time.time()
        files = []
        if self.render_dir.exists():
            files = [
                f for f in self.render_dir.iterdir()
                if f.is_file() and f.name.startswith("fav_") and f.suffix == ".png"
            ]
        stats = []
        for file in files:
            try:
                stats.append(file.stat())
            except OSError:
                pass
        return {
            "count": len(stats),
            "size_bytes": sum(item.st_size for item in stats),
            "oldest_seconds": int(max((now - item.st_mtime for item in stats), default=0)),
            "dir": str(self.render_dir),
        }

    def _save_img(self, img: "Image.Image") -> str:
        path = self.render_dir / f"fav_{uuid.uuid4().hex[:12]}.png"
        # 保留圆角卡片外侧与投影区域的 Alpha 通道，让聊天客户端
        # 能直接叠加在任意会话背景上，而不是显示为黑色矩形底。
        img.convert("RGBA").save(path, format="PNG", optimize=True)
        return str(path)

    @staticmethod
    def _gradient(size: tuple[int, int], top: str, bottom: str) -> "Image.Image":
        width, height = size
        top_rgb, bottom_rgb = _hex(top), _hex(bottom)
        strip = Image.new("RGB", (1, max(height, 1)))
        for y in range(max(height, 1)):
            strip.putpixel((0, y), _mix(top_rgb, bottom_rgb, y / max(height - 1, 1)))
        return strip.resize((width, height)).convert("RGBA")

    @staticmethod
    def _radial_glow(size: tuple[int, int], color: tuple[int, int, int], alpha: int) -> "Image.Image":
        width, height = size
        glow_size = max(width, height)
        dot = Image.new("RGBA", (glow_size, glow_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dot)
        radius = glow_size // 3
        cx, cy = glow_size // 2, glow_size // 2
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(*color, alpha))
        return dot.filter(ImageFilter.GaussianBlur(max(30, radius // 2)))

    def _base_card(self, width: int, height: int, accent: tuple[int, int, int]) -> "Image.Image":
        theme = self.theme
        margin = 24
        canvas = Image.new("RGBA", (width + margin * 2, height + margin * 2), (0, 0, 0, 0))
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle(
            (margin + 8, margin + 10, margin + width - 8, margin + height - 5),
            radius=32,
            fill=(0, 0, 0, theme["shadow"]),
        )
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(16)))

        card = self._gradient((width, height), theme["top"], theme["bottom"])
        glow = self._radial_glow((width, height), accent, theme["glow"])
        card.alpha_composite(glow, (width - glow.width // 2, -glow.height // 2))
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=30, fill=255)
        card.putalpha(mask)
        canvas.alpha_composite(card, (margin, margin))

        border = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(border).rounded_rectangle(
            (margin, margin, margin + width - 1, margin + height - 1),
            radius=30,
            outline=_rgba(theme["border"], theme["border_alpha"]),
            width=1,
        )
        canvas.alpha_composite(border)
        return canvas

    def _surface(
        self,
        canvas: "Image.Image",
        box: tuple[int, int, int, int],
        *,
        radius: int = 18,
        strong: bool = False,
        accent: tuple[int, int, int] | None = None,
    ) -> None:
        theme = self.theme
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        alpha = theme["surface_strong"] if strong else theme["surface_alpha"]
        fill = accent if accent else _hex(theme["surface"])
        ImageDraw.Draw(layer).rounded_rectangle(
            box,
            radius=radius,
            fill=(*fill, alpha if not accent else max(alpha, 18)),
            outline=_rgba(theme["border"], theme["border_alpha"]),
            width=1,
        )
        canvas.alpha_composite(layer)

    @staticmethod
    def _text_width(draw: "ImageDraw.ImageDraw", text: str, font: "ImageFont.FreeTypeFont") -> int:
        return math.ceil(draw.textlength(text, font=font))

    def _ellipsize(self, draw, text: str, font, max_width: int) -> str:
        text = str(text or "")
        if self._text_width(draw, text, font) <= max_width:
            return text
        while text and self._text_width(draw, text + "…", font) > max_width:
            text = text[:-1]
        return text + "…"

    def _fit_font(
        self,
        draw,
        text: str,
        max_width: int,
        preferred_size: int,
        min_size: int,
        *,
        bold: bool = False,
    ):
        """在指定宽度内选择尽可能大的字号，避免长数字侵入相邻内容。"""
        for size in range(preferred_size, min_size - 1, -1):
            font = _load_font(size, bold)
            if self._text_width(draw, text, font) <= max_width:
                return font
        return _load_font(min_size, bold)

    def _wrap(self, draw, text: str, font, max_width: int, max_lines: int) -> list[str]:
        lines: list[str] = []
        current = ""
        for char in str(text or "暂无评价"):
            if char == "\n":
                lines.append(current)
                current = ""
            elif self._text_width(draw, current + char, font) <= max_width:
                current += char
            else:
                lines.append(current)
                current = char
            if len(lines) == max_lines:
                break
        if len(lines) < max_lines and current:
            lines.append(current)
        if len(lines) == max_lines:
            consumed = "".join(lines)
            raw = str(text or "暂无评价").replace("\n", "")
            if len(consumed) < len(raw):
                lines[-1] = self._ellipsize(draw, lines[-1] + raw[len(consumed):], font, max_width)
        return lines or ["暂无评价"]

    @staticmethod
    def _center_text(draw, box, text, font, fill) -> None:
        x0, y0, x1, y1 = box
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - bbox[1]), text, font=font, fill=fill)

    def render_favorability_card(
        self,
        user_name: str,
        user_id: str,
        score: int,
        evaluation: str,
        relation: str = "",
    ) -> str:
        """渲染个人好感度档案卡。徽章与配色由关系档位决定（与分数解耦）。"""
        width, height, margin = 720, 420, 24
        level = get_relation_info(relation) if relation else get_level_info(score)
        accent = _hex(level["color"])
        canvas = self._base_card(width, height, accent)
        draw = ImageDraw.Draw(canvas)
        theme = self.theme
        ox, oy = margin, margin

        # 顶部品牌短横条与标签（徽章宽度随关系名自适应）
        draw.rounded_rectangle((ox + 42, oy + 34, ox + 108, oy + 40), radius=3, fill=accent)
        badge_font = _load_font(18, True)
        badge_w = self._text_width(draw, level["title"], badge_font) + 28
        badge_box = (ox + 42, oy + 60, ox + 42 + badge_w, oy + 96)
        self._surface(canvas, badge_box, radius=18, accent=accent)
        self._center_text(draw, badge_box, level["title"], badge_font, accent)
        draw.text(
            (badge_box[2] + 14, oy + 67),
            "FAVORABILITY PROFILE",
            font=_load_font(15, True),
            fill=_hex(theme["tertiary"]),
        )

        # 用户信息
        name_font = _load_font(34, True)
        name = self._ellipsize(draw, user_name or f"用户 {user_id}", name_font, 390)
        draw.text((ox + 42, oy + 118), name, font=name_font, fill=_hex(theme["text"]))
        id_font = _load_font(17)
        draw.text((ox + 43, oy + 164), f"ID  ·  {user_id}", font=id_font, fill=_hex(theme["tertiary"]))

        # 分数视觉焦点
        score_text = f"{score:+d}" if score else "0"
        score_font = self._fit_font(draw, score_text, 260, 68, 28, bold=True)
        score_w = self._text_width(draw, score_text, score_font)
        draw.text((ox + width - 48 - score_w, oy + 78), score_text, font=score_font, fill=accent)
        label_font = _load_font(16, True)
        label = "好 感 度"
        label_w = self._text_width(draw, label, label_font)
        draw.text((ox + width - 49 - label_w, oy + 158), label, font=label_font, fill=_hex(theme["tertiary"]))

        # 评价区
        panel = (ox + 42, oy + 210, ox + width - 42, oy + 342)
        self._surface(canvas, panel, radius=22, strong=True)
        draw.rounded_rectangle((panel[0] + 18, panel[1] + 22, panel[0] + 24, panel[3] - 22), radius=3, fill=accent)
        draw.text((panel[0] + 42, panel[1] + 20), "她的评价", font=_load_font(16, True), fill=_hex(theme["tertiary"]))
        eval_font = _load_font(22)
        for index, line in enumerate(self._wrap(draw, evaluation, eval_font, panel[2] - panel[0] - 76, 2)):
            draw.text((panel[0] + 42, panel[1] + 51 + index * 32), line, font=eval_font, fill=_hex(theme["text"]))

        # 页脚说明与水印
        draw.text((ox + 43, oy + 369), level["description"], font=_load_font(17), fill=_hex(theme["secondary"]))
        wm = "FAVORABILITY"
        wm_font = _load_font(14, True)
        wm_w = self._text_width(draw, wm, wm_font)
        draw.ellipse((ox + width - 48 - wm_w - 18, oy + 375, ox + width - 40 - wm_w, oy + 383), fill=accent)
        draw.text((ox + width - 34 - wm_w, oy + 369), wm, font=wm_font, fill=_hex(theme["tertiary"]))
        return self._save_img(canvas)

    def render_ranking_image(
        self,
        ranked_list: list[tuple[str, dict]],
        group_name: str = "",
        ascending: bool = False,
    ) -> str:
        """渲染好感度排行卡片。"""
        if not ranked_list:
            return self.render_empty_ranking()
        rows = ranked_list[:10]
        width, margin = 760, 24
        row_h = 68
        height = 224 + len(rows) * row_h + 76
        accent = _hex("#8B7CF6" if ascending else "#FF6B8A")
        canvas = self._base_card(width, height, accent)
        draw = ImageDraw.Draw(canvas)
        theme = self.theme
        ox, oy = margin, margin

        draw.rounded_rectangle((ox + 42, oy + 34, ox + 108, oy + 40), radius=3, fill=accent)
        title = "好感度低谷榜" if ascending else "好感度排行榜"
        draw.text((ox + 42, oy + 64), title, font=_load_font(34, True), fill=_hex(theme["text"]))
        subtitle = f"{group_name + '  ·  ' if group_name else ''}共 {len(rows)} 位用户  ·  {'由低到高' if ascending else '由高到低'}"
        draw.text((ox + 43, oy + 112), subtitle, font=_load_font(17), fill=_hex(theme["tertiary"]))

        # 概览胶囊
        best_score = int(rows[0][1].get("score", 0))
        overview = (ox + width - 222, oy + 54, ox + width - 42, oy + 126)
        self._surface(canvas, overview, radius=20, strong=True, accent=accent)
        draw.text((overview[0] + 20, overview[1] + 13), "当前榜首", font=_load_font(15, True), fill=_hex(theme["tertiary"]))
        score_label = f"{best_score:+d}" if best_score else "0"
        level_title = get_relation_info(rows[0][1].get("relation"))["title"]
        level_font = _load_font(15, True)
        level_w = self._text_width(draw, level_title, level_font)
        draw.text(
            (overview[2] - 20 - level_w, overview[1] + 13),
            level_title,
            font=level_font,
            fill=_hex(theme["secondary"]),
        )
        overview_score_font = self._fit_font(
            draw, score_label, overview[2] - overview[0] - 40, 27, 14, bold=True
        )
        draw.text(
            (overview[0] + 20, overview[1] + 37),
            score_label,
            font=overview_score_font,
            fill=accent,
        )

        # 表头
        header_y = oy + 158
        self._surface(canvas, (ox + 42, header_y, ox + width - 42, header_y + 46), radius=15, strong=True)
        headers = [(ox + 62, "排名"), (ox + 150, "昵称"), (ox + 430, "评价"), (ox + 650, "分数")]
        for x, text in headers:
            draw.text((x, header_y + 13), text, font=_load_font(15, True), fill=_hex(theme["tertiary"]))

        medals = ["01", "02", "03"]
        medal_colors = [_hex("#FFB547"), _hex("#9DA9BE"), _hex("#C98561")]
        medal_text_colors = [_hex("#3B2A08"), _hex("#FFFFFF"), _hex("#FFFFFF")]
        y = header_y + 54
        for index, (uid, data) in enumerate(rows):
            score = int(data.get("score", 0))
            relation = str(data.get("relation") or "") or relation_for_score(score)
            level_color = _hex(get_relation_info(relation)["color"])
            row_box = (ox + 42, y, ox + width - 42, y + row_h - 8)
            if index < 3:
                self._surface(canvas, row_box, radius=17, strong=True, accent=medal_colors[index])
            elif index % 2 == 0:
                self._surface(canvas, row_box, radius=17)

            rank_box = (ox + 58, y + 13, ox + 100, y + 43)
            rank_color = medal_colors[index] if index < 3 else _hex(theme["tertiary"])
            if index < 3:
                # 前三名序号使用完全不透明的实色底，避免客户端合成透明 PNG
                # 时出现底色发灰、文字对比度不稳定等问题。
                draw.rounded_rectangle(
                    rank_box,
                    radius=15,
                    fill=(*rank_color, 255),
                )
            rank_text_color = (
                medal_text_colors[index] if index < 3 else rank_color
            )
            self._center_text(
                draw,
                rank_box,
                medals[index] if index < 3 else f"{index + 1:02d}",
                _load_font(15, True),
                rank_text_color,
            )

            name_font = _load_font(19, True)
            name = str(data.get("name") or uid)
            draw.text((ox + 150, y + 13), self._ellipsize(draw, name, name_font, 245), font=name_font, fill=_hex(theme["text"]))
            eval_font = _load_font(16)
            evaluation = self._ellipsize(draw, str(data.get("eval") or "暂无评价"), eval_font, 176)
            draw.text((ox + 430, y + 16), evaluation, font=eval_font, fill=_hex(theme["secondary"]))
            score_text = f"{score:+d}" if score else "0"
            score_font = self._fit_font(draw, score_text, 82, 20, 12, bold=True)
            score_w = self._text_width(draw, score_text, score_font)
            draw.text((ox + width - 54 - score_w, y + 12), score_text, font=score_font, fill=level_color)
            y += row_h

        divider_y = oy + height - 64
        draw.line((ox + 42, divider_y, ox + width - 42, divider_y), fill=_rgba(theme["border"], 18), width=1)
        draw.text((ox + 43, divider_y + 22), "发送「查询好感度」查看你的详细档案", font=_load_font(16), fill=_hex(theme["tertiary"]))
        wm = "FAVORABILITY RANK"
        wm_font = _load_font(14, True)
        wm_w = self._text_width(draw, wm, wm_font)
        draw.ellipse((ox + width - 49 - wm_w - 18, divider_y + 27, ox + width - 41 - wm_w, divider_y + 35), fill=accent)
        draw.text((ox + width - 35 - wm_w, divider_y + 21), wm, font=wm_font, fill=_hex(theme["tertiary"]))
        return self._save_img(canvas)

    def render_empty_ranking(self) -> str:
        width, height, margin = 620, 280, 24
        accent = _hex("#8B7CF6")
        canvas = self._base_card(width, height, accent)
        draw = ImageDraw.Draw(canvas)
        theme = self.theme
        ox, oy = margin, margin
        draw.rounded_rectangle((ox + 42, oy + 34, ox + 108, oy + 40), radius=3, fill=accent)
        draw.text((ox + 42, oy + 66), "好感度排行榜", font=_load_font(30, True), fill=_hex(theme["text"]))
        panel = (ox + 42, oy + 128, ox + width - 42, oy + 222)
        self._surface(canvas, panel, radius=22, strong=True, accent=accent)
        draw.ellipse((panel[0] + 24, panel[1] + 27, panel[0] + 64, panel[1] + 67), fill=(*accent, 30), outline=(*accent, 90), width=1)
        self._center_text(draw, (panel[0] + 24, panel[1] + 27, panel[0] + 64, panel[1] + 67), "—", _load_font(20, True), accent)
        draw.text((panel[0] + 82, panel[1] + 22), "暂无好感度记录", font=_load_font(21, True), fill=_hex(theme["text"]))
        draw.text((panel[0] + 82, panel[1] + 52), "与 AI 开始聊天后，这里会自动生成排行", font=_load_font(16), fill=_hex(theme["tertiary"]))
        return self._save_img(canvas)
