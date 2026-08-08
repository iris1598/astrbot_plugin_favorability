"""Modern Pillow renderers for favorability cards and rankings.

The visual language mirrors the two Rika Share themes: a rounded gradient card,
subtle accent glow, soft shadow, and frosted-glass content surfaces.  Both
``dark`` and ``light`` themes use the same layout so switching themes only
changes presentation, never the information shown to users.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from astrbot.api import logger

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    HAS_PIL = True
except ImportError:  # pragma: no cover - depends on optional dependency
    HAS_PIL = False
    Image = ImageDraw = ImageFilter = ImageFont = None
    logger.warning(
        "[favorability] Pillow is not installed; image rendering is unavailable."
    )


_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/deng.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
)


@lru_cache(maxsize=32)
def _load_font(size: int, bold: bool = False):
    """Load an available CJK-capable font and fall back to Pillow's default."""
    candidates = _FONT_CANDIDATES
    if bold:
        candidates = (
            tuple(
                path
                for path in candidates
                if "bd" in path.lower() or "bold" in path.lower()
            )
            + candidates
        )
    for path in candidates:
        if not Path(path).is_file():
            continue
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Convert a CSS hexadecimal colour to an RGB tuple."""
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _with_alpha(rgb: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return (*rgb, alpha)


def _mix(
    first: tuple[int, int, int], second: tuple[int, int, int], ratio: float
) -> tuple[int, int, int]:
    """Mix two RGB colours using a ratio between zero and one."""
    return tuple(
        round(first[index] + (second[index] - first[index]) * ratio)
        for index in range(3)
    )


@dataclass(frozen=True)
class _Theme:
    gradient_top: tuple[int, int, int]
    gradient_bottom: tuple[int, int, int]
    border: tuple[int, int, int]
    text_primary: tuple[int, int, int]
    text_secondary: tuple[int, int, int]
    text_tertiary: tuple[int, int, int]
    frost: tuple[int, int, int]
    shadow_alpha: int
    glow_alpha: int
    frost_alpha: int
    frost_border_alpha: int
    border_alpha: int


# These values intentionally match the dark and light colour language used by
# astrbot_plugin_rika_share's modern sharing cards.
_THEMES = {
    "dark": _Theme(
        gradient_top=_hex_to_rgb("#242B3F"),
        gradient_bottom=_hex_to_rgb("#12161F"),
        border=_hex_to_rgb("#FFFFFF"),
        text_primary=_hex_to_rgb("#F5F7FC"),
        text_secondary=_hex_to_rgb("#AEB6C8"),
        text_tertiary=_hex_to_rgb("#7B8598"),
        frost=_hex_to_rgb("#FFFFFF"),
        shadow_alpha=130,
        glow_alpha=30,
        frost_alpha=14,
        frost_border_alpha=26,
        border_alpha=24,
    ),
    "light": _Theme(
        gradient_top=_hex_to_rgb("#FFFFFF"),
        gradient_bottom=_hex_to_rgb("#F1F4F9"),
        border=_hex_to_rgb("#1B2233"),
        text_primary=_hex_to_rgb("#1A2130"),
        text_secondary=_hex_to_rgb("#55607A"),
        text_tertiary=_hex_to_rgb("#8C95A9"),
        frost=_hex_to_rgb("#1B2233"),
        shadow_alpha=55,
        glow_alpha=16,
        frost_alpha=10,
        frost_border_alpha=20,
        border_alpha=14,
    ),
}


def get_level_info(score: int) -> dict[str, str]:
    """Return the favorability label, accent colour, and description."""
    if score >= 70:
        return {
            "title": "挚爱",
            "color": "#FF4757",
            "bg": "#FFE8EA",
            "description": "爱人级，关系已不分彼此",
        }
    if score >= 50:
        return {
            "title": "挚友",
            "color": "#FF6348",
            "bg": "#FFE4DB",
            "description": "挚友 / 恋人级，热情主动",
        }
    if score >= 21:
        return {
            "title": "熟人",
            "color": "#2ED573",
            "bg": "#E4FCEF",
            "description": "熟人级，积极友好",
        }
    if score >= -20:
        return {
            "title": "路人",
            "color": "#747D8C",
            "bg": "#F0F0F1",
            "description": "陌生人级，礼貌中立",
        }
    if score >= -50:
        return {
            "title": "生厌",
            "color": "#FFA502",
            "bg": "#FFF3E0",
            "description": "反感和警惕，厌恶这类行为",
        }
    if score >= -70:
        return {
            "title": "憎恶",
            "color": "#FF4757",
            "bg": "#FFEBEE",
            "description": "极度厌恶，会直接表达不满",
        }
    return {
        "title": "仇敌",
        "color": "#2F3542",
        "bg": "#E8E8E8",
        "description": "光是看到就令人不快",
    }


def _format_score(score: int, max_digits: int = 12) -> str:
    """Keep unbounded scores legible without changing their stored value.

    Scores within ``max_digits`` render exactly. Larger values use scientific
    notation only in the image, while storage and ranking continue to use the
    original integer.
    """
    text = f"{score:+d}" if score else "0"
    if len(text) <= max_digits:
        return text
    digits = str(abs(score))
    sign = "-" if score < 0 else "+"
    return f"{sign}{digits[0]}.{digits[1:4]}e+{len(digits) - 1}"


class FavorabilityRenderer:
    """Render favorability information with a selectable Rika-style theme."""

    CARD_RADIUS = 30
    CARD_SHADOW_INSET = 10
    CARD_SHADOW_BLUR = 16

    def __init__(
        self,
        render_dir: str | Path,
        cache_max_age: int = 3600,
        theme: str = "dark",
    ):
        """Create a renderer.

        Args:
            render_dir: Directory used for temporary rendered images.
            cache_max_age: Maximum cache lifetime in seconds.
            theme: Either ``dark`` or ``light``. Invalid values use ``dark``.
        """
        if not HAS_PIL:
            raise RuntimeError("Pillow is required for favorability image rendering")
        self.render_dir = Path(render_dir)
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self.cache_max_age = cache_max_age
        self.theme_name = theme if theme in _THEMES else "dark"

    @property
    def theme(self) -> _Theme:
        """Return the currently selected visual theme."""
        return _THEMES[self.theme_name]

    def cleanup_cache(self, max_age: int | None = None) -> tuple[int, int]:
        """Delete expired favorability render-cache images.

        Args:
            max_age: Cache lifetime in seconds. Defaults to ``cache_max_age``.

        Returns:
            A tuple of deleted and remaining image counts.
        """
        max_age = self.cache_max_age if max_age is None else max_age
        now = time.time()
        deleted = 0
        remaining = 0
        for path in self.render_dir.glob("fav_*.png"):
            try:
                if now - path.stat().st_mtime > max_age:
                    path.unlink()
                    deleted += 1
                else:
                    remaining += 1
            except OSError:
                remaining += 1
        return deleted, remaining

    def get_cache_info(self) -> dict[str, int | str]:
        """Return cache statistics for the admin cleanup command."""
        paths = list(self.render_dir.glob("fav_*.png"))
        now = time.time()
        size_bytes = 0
        oldest_seconds = 0
        for path in paths:
            try:
                stat = path.stat()
                size_bytes += stat.st_size
                oldest_seconds = max(oldest_seconds, int(now - stat.st_mtime))
            except OSError:
                continue
        return {
            "count": len(paths),
            "size_bytes": size_bytes,
            "oldest_seconds": oldest_seconds,
            "dir": str(self.render_dir),
        }

    def _save_img(self, image) -> str:
        path = self.render_dir / f"fav_{uuid.uuid4().hex[:12]}.png"
        image.save(path, format="PNG")
        return str(path)

    @staticmethod
    def _gradient(
        size: tuple[int, int], top: tuple[int, int, int], bottom: tuple[int, int, int]
    ):
        width, height = size
        gradient = Image.new("RGB", (1, max(height, 1)))
        for y in range(max(height, 1)):
            ratio = y / max(height - 1, 1)
            gradient.putpixel((0, y), _mix(top, bottom, ratio))
        return gradient.resize((width, height))

    @staticmethod
    def _radial_glow(width: int, height: int, accent: tuple[int, int, int], alpha: int):
        glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse(
            (-width // 3, -height // 2, width // 2, height // 2),
            fill=_with_alpha(accent, alpha),
        )
        return glow.filter(ImageFilter.GaussianBlur(max(width, height) // 5))

    def _base_canvas(self, width: int, card_height: int, accent: tuple[int, int, int]):
        """Create the shared gradient, glow, shadow, and rounded-card layer."""
        total_height = card_height + 14
        theme = self.theme
        canvas = Image.new("RGBA", (width, total_height), (0, 0, 0, 0))
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle(
            (
                self.CARD_SHADOW_INSET,
                self.CARD_SHADOW_INSET,
                width - self.CARD_SHADOW_INSET,
                total_height - 2,
            ),
            radius=self.CARD_RADIUS + 2,
            fill=(0, 0, 0, theme.shadow_alpha),
        )
        canvas.alpha_composite(
            shadow.filter(ImageFilter.GaussianBlur(self.CARD_SHADOW_BLUR))
        )

        card = self._gradient(
            (width, card_height), theme.gradient_top, theme.gradient_bottom
        ).convert("RGBA")
        card.alpha_composite(
            self._radial_glow(width, round(width * 0.9), accent, theme.glow_alpha)
        )
        mask = Image.new("L", (width, card_height), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, width - 1, card_height - 1), radius=self.CARD_RADIUS, fill=255
        )
        card.putalpha(mask)
        canvas.alpha_composite(card)

        border = Image.new("RGBA", (width, card_height), (0, 0, 0, 0))
        ImageDraw.Draw(border).rounded_rectangle(
            (0, 0, width - 1, card_height - 1),
            radius=self.CARD_RADIUS,
            outline=_with_alpha(theme.border, theme.border_alpha),
            width=1,
        )
        canvas.alpha_composite(border)
        return canvas

    def _glass(
        self,
        canvas,
        box: tuple[int, int, int, int],
        radius: int,
        *,
        tint: tuple[int, int, int] | None = None,
        tint_alpha: int | None = None,
        border: tuple[int, int, int] | None = None,
        border_alpha: int | None = None,
    ) -> None:
        """Draw a frosted-glass rounded surface on an existing canvas."""
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            return
        theme = self.theme
        region = canvas.crop(box).filter(ImageFilter.GaussianBlur(6))
        tint = theme.frost if tint is None else tint
        tint_alpha = theme.frost_alpha if tint_alpha is None else tint_alpha
        border = theme.frost if border is None else border
        border_alpha = (
            theme.frost_border_alpha if border_alpha is None else border_alpha
        )
        region.alpha_composite(
            Image.new("RGBA", region.size, _with_alpha(tint, tint_alpha))
        )
        mask = Image.new("L", region.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, region.width - 1, region.height - 1), radius=radius, fill=255
        )
        canvas.paste(region, (x0, y0), mask)

        if border_alpha:
            border_layer = Image.new("RGBA", region.size, (0, 0, 0, 0))
            ImageDraw.Draw(border_layer).rounded_rectangle(
                (0, 0, region.width - 1, region.height - 1),
                radius=radius,
                outline=_with_alpha(border, border_alpha),
                width=1,
            )
            canvas.alpha_composite(border_layer, (x0, y0))

    @staticmethod
    def _line_height(font) -> int:
        ascent, descent = font.getmetrics()
        return ascent + descent

    @staticmethod
    def _text_width(draw, text: str, font) -> int:
        return math.ceil(draw.textlength(text, font=font))

    def _truncate(self, draw, text: str, font, max_width: int) -> str:
        """Fit a single-line string in the given width, adding an ellipsis."""
        if self._text_width(draw, text, font) <= max_width:
            return text
        shortened = text
        while shortened and self._text_width(draw, shortened + "…", font) > max_width:
            shortened = shortened[:-1]
        return f"{shortened}…" if shortened else "…"

    def _wrap(self, draw, text: str, font, max_width: int, max_lines: int) -> list[str]:
        """Wrap CJK-friendly text by glyph width and cap it at ``max_lines``."""
        text = text or "暂无"
        lines: list[str] = []
        for paragraph in text.replace("\r", "").split("\n"):
            current = ""
            for char in paragraph or " ":
                if current and self._text_width(draw, current + char, font) > max_width:
                    lines.append(current)
                    current = char
                else:
                    current += char
            if current:
                lines.append(current)
        if len(lines) <= max_lines:
            return lines
        return lines[: max_lines - 1] + [
            self._truncate(draw, lines[max_lines - 1], font, max_width)
        ]

    def _draw_pill(
        self,
        canvas,
        x: int,
        y: int,
        text: str,
        accent: tuple[int, int, int],
        *,
        height: int = 40,
        text_color: tuple[int, int, int] | None = None,
        dot: bool = True,
    ) -> int:
        """Draw a glass pill and return its width."""
        draw = ImageDraw.Draw(canvas)
        font = _load_font(18, bold=True)
        text_color = self.theme.text_primary if text_color is None else text_color
        width = self._text_width(draw, text, font) + 34 + (20 if dot else 0)
        self._glass(canvas, (x, y, x + width, y + height), height // 2)
        text_x = x + 17
        if dot:
            dot_y = y + height // 2
            draw.ellipse((text_x, dot_y - 5, text_x + 10, dot_y + 5), fill=accent)
            text_x += 18
        draw.text(
            (text_x, y + (height - self._line_height(font)) // 2),
            text,
            font=font,
            fill=text_color,
        )
        return width

    def _draw_divider(self, canvas, x0: int, x1: int, y: int) -> None:
        layer = Image.new("RGBA", (x1 - x0, 1), (0, 0, 0, 0))
        ImageDraw.Draw(layer).line(
            (0, 0, x1 - x0, 0),
            fill=_with_alpha(
                self.theme.border, 26 if self.theme_name == "dark" else 20
            ),
        )
        canvas.alpha_composite(layer, (x0, y))

    def _draw_footer(self, canvas, card_width: int, y: int, accent) -> None:
        """Draw the small branded footer shared by all rendered images."""
        draw = ImageDraw.Draw(canvas)
        font = _load_font(17, bold=True)
        text = "好感度系统"
        text_width = self._text_width(draw, text, font)
        dot_x = card_width - 44 - text_width - 18
        dot_y = y + self._line_height(font) // 2
        draw.ellipse((dot_x, dot_y - 5, dot_x + 10, dot_y + 5), fill=accent)
        draw.text((dot_x + 18, y), text, font=font, fill=accent)

    def render_favorability_card(
        self, user_name: str, user_id: str, score: int, evaluation: str
    ) -> str:
        """Render a compact two-column favorability profile card."""
        width, card_height, pad = 720, 446, 44
        level = get_level_info(score)
        accent = _hex_to_rgb(level["color"])
        if self.theme_name == "dark" and sum(accent) < 260:
            accent = _mix(accent, (255, 255, 255), 0.45)
        canvas = self._base_canvas(width, card_height, accent)
        draw = ImageDraw.Draw(canvas)
        theme = self.theme

        # Keep the original compact Rika-style composition: identity and score
        # at the top, relationship status and impression side by side below.
        bar = Image.new("RGBA", (64, 6), (0, 0, 0, 0))
        for x in range(64):
            ImageDraw.Draw(bar).line(
                (x, 0, x, 5),
                fill=_with_alpha(_mix(accent, (255, 255, 255), x / 100), 255),
            )
        canvas.alpha_composite(bar, (pad, 34))
        self._draw_pill(canvas, pad, 58, "好感度档案", accent)

        name_font = _load_font(38, bold=True)
        id_font = _load_font(19)
        body_font = _load_font(19)
        small_font = _load_font(16)

        display_name = self._truncate(draw, user_name or "未知用户", name_font, 365)
        draw.text((pad, 121), display_name, font=name_font, fill=theme.text_primary)
        draw.text((pad, 174), f"ID  {user_id}", font=id_font, fill=theme.text_tertiary)

        score_caption_font = _load_font(18, bold=True)
        score_text = _format_score(score)
        score_size = 74
        while score_size > 32:
            score_font = _load_font(score_size, bold=True)
            if self._text_width(draw, score_text, score_font) <= 220:
                break
            score_size -= 4
        score_font = _load_font(score_size, bold=True)
        score_width = self._text_width(draw, score_text, score_font)
        score_x = width - pad - score_width
        draw.text(
            (score_x, 95),
            score_text,
            font=score_font,
            fill=accent,
            stroke_width=1,
            stroke_fill=_mix(accent, theme.gradient_bottom, 0.55),
        )
        draw.text(
            (width - pad - self._text_width(draw, "好感度", score_caption_font), 176),
            "好感度",
            font=score_caption_font,
            fill=theme.text_tertiary,
        )
        self._draw_divider(canvas, pad, width - pad, 218)

        left_x, panel_y, left_w, panel_h = pad, 246, 256, 142
        status_x = left_x
        draw.text(
            (status_x, panel_y + 18),
            "当前关系",
            font=small_font,
            fill=theme.text_tertiary,
        )
        self._draw_pill(
            canvas, status_x, panel_y + 43, level["title"], accent, height=42
        )
        draw.text(
            (status_x, panel_y + 103),
            level["description"],
            font=small_font,
            fill=theme.text_secondary,
        )

        # Evaluation remains the calm, frosted-glass counterpart to the status.
        eval_x, eval_y, eval_w, eval_h = left_x + left_w + 34, panel_y, 342, panel_h
        self._glass(
            canvas,
            (eval_x, eval_y, eval_x + eval_w, eval_y + eval_h),
            20,
            border=accent,
            border_alpha=46 if self.theme_name == "dark" else 38,
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (eval_x + 22, eval_y + 18),
            "她对你的印象",
            font=small_font,
            fill=theme.text_tertiary,
        )
        eval_lines = self._wrap(draw, evaluation, body_font, eval_w - 44, 2)
        line_y = eval_y + 48
        for line in eval_lines:
            draw.text(
                (eval_x + 22, line_y), line, font=body_font, fill=theme.text_primary
            )
            line_y += self._line_height(body_font) + 5

        self._draw_divider(canvas, pad, width - pad, 404)
        self._draw_footer(canvas, width, 412, accent)
        return self._save_img(canvas)

    def render_ranking_image(
        self,
        ranked_list: list[tuple[str, dict]],
        group_name: str = "",
        ascending: bool = False,
    ) -> str:
        """Render a themed ranking image, keeping the existing public API."""
        if not ranked_list:
            return self.render_empty_ranking()

        width, pad, row_height, row_gap = 720, 40, 80, 8
        title = "好感度排行榜"
        order_text = "由低到高" if ascending else "由高到低"
        subtitle = f"当前会话 · 共 {len(ranked_list)} 位用户"
        row_y = 192
        footer_y = row_y + len(ranked_list) * (row_height + row_gap) - row_gap + 16
        card_height = footer_y + 88
        accent = _hex_to_rgb("#8B7CF6")
        canvas = self._base_canvas(width, card_height, accent)
        draw = ImageDraw.Draw(canvas)
        theme = self.theme

        bar = Image.new("RGBA", (64, 6), (0, 0, 0, 0))
        for x in range(64):
            ImageDraw.Draw(bar).line(
                (x, 0, x, 5),
                fill=_with_alpha(_mix(accent, (255, 255, 255), x / 100), 255),
            )
        canvas.alpha_composite(bar, (pad, 32))
        self._draw_pill(canvas, pad, 54, "当前会话", accent)
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (pad, 106), title, font=_load_font(31, bold=True), fill=theme.text_primary
        )
        draw.text((pad, 147), subtitle, font=_load_font(17), fill=theme.text_tertiary)
        order_font = _load_font(16, bold=True)
        order_width = self._text_width(draw, order_text, order_font) + 54
        self._draw_pill(
            canvas,
            width - pad - order_width,
            112,
            order_text,
            accent,
            height=36,
            dot=False,
            text_color=theme.text_secondary,
        )
        self._draw_divider(canvas, pad, width - pad, 172)

        medal_colours = ("#F6B83F", "#A7B3C4", "#D89C68")
        for index, (user_id, user_data) in enumerate(ranked_list):
            score = int(user_data.get("score", 0))
            level = get_level_info(score)
            level_accent = _hex_to_rgb(level["color"])
            if self.theme_name == "dark" and sum(level_accent) < 260:
                level_accent = _mix(level_accent, (255, 255, 255), 0.45)
            self._glass(canvas, (pad, row_y, width - pad, row_y + row_height), 18)
            draw = ImageDraw.Draw(canvas)

            rank_x = pad + 14
            rank_y = row_y + 22
            rank_colour = (
                _hex_to_rgb(medal_colours[index]) if index < 3 else theme.text_tertiary
            )
            rank_layer = Image.new("RGBA", (36, 36), (0, 0, 0, 0))
            ImageDraw.Draw(rank_layer).ellipse(
                (0, 0, 35, 35), fill=_with_alpha(rank_colour, 220)
            )
            canvas.alpha_composite(rank_layer, (rank_x, rank_y))
            rank_font = _load_font(15, bold=True)
            rank_text = f"{index + 1:02d}"
            rank_width = self._text_width(draw, rank_text, rank_font)
            draw.text(
                (rank_x + (36 - rank_width) // 2, rank_y + 8),
                rank_text,
                font=rank_font,
                fill=(255, 255, 255),
            )

            primary_y = row_y + 15
            secondary_y = row_y + 48
            name = str(user_data.get("name") or user_id)
            name_font = _load_font(20, bold=True)
            eval_font = _load_font(15)
            display_name = self._truncate(draw, name, name_font, 254)
            evaluation = self._truncate(
                draw, str(user_data.get("eval") or "暂无评价"), eval_font, 254
            )
            draw.text(
                (pad + 70, primary_y),
                display_name,
                font=name_font,
                fill=theme.text_primary,
            )
            draw.text(
                (pad + 70, secondary_y),
                evaluation,
                font=eval_font,
                fill=theme.text_tertiary,
            )

            score_text = _format_score(score)
            score_size = 27
            while score_size > 17:
                score_font = _load_font(score_size, bold=True)
                if self._text_width(draw, score_text, score_font) <= 138:
                    break
                score_size -= 2
            score_font = _load_font(score_size, bold=True)
            score_width = self._text_width(draw, score_text, score_font)
            score_x = width - pad - 22 - score_width
            draw.text(
                (score_x, primary_y), score_text, font=score_font, fill=level_accent
            )
            status_font = _load_font(15, bold=True)
            status_x = pad + 356
            status_y = secondary_y
            draw.ellipse(
                (status_x, status_y + 5, status_x + 8, status_y + 13),
                fill=level_accent,
            )
            draw.text(
                (status_x + 16, status_y),
                level["title"],
                font=status_font,
                fill=theme.text_secondary,
            )
            row_y += row_height + row_gap

        self._glass(canvas, (pad, footer_y, width - pad, footer_y + 42), 16)
        draw = ImageDraw.Draw(canvas)
        hint = "发送「查询好感度」查看你的详细档案"
        hint_font = _load_font(16)
        hint_width = self._text_width(draw, hint, hint_font)
        draw.text(
            ((width - hint_width) // 2, footer_y + 11),
            hint,
            font=hint_font,
            fill=theme.text_secondary,
        )
        self._draw_footer(canvas, width, footer_y + 54, accent)
        return self._save_img(canvas)

    def render_empty_ranking(self) -> str:
        """Render the themed empty state used when a ranking has no records."""
        width, card_height, pad = 600, 306, 42
        accent = _hex_to_rgb("#8B7CF6")
        canvas = self._base_canvas(width, card_height, accent)
        draw = ImageDraw.Draw(canvas)
        theme = self.theme
        bar = Image.new("RGBA", (64, 6), (0, 0, 0, 0))
        for x in range(64):
            ImageDraw.Draw(bar).line(
                (x, 0, x, 5),
                fill=_with_alpha(_mix(accent, (255, 255, 255), x / 100), 255),
            )
        canvas.alpha_composite(bar, (pad, 34))
        self._draw_pill(canvas, pad, 58, "当前会话", accent)
        self._glass(canvas, (pad, 122, width - pad, 235), 22)
        draw = ImageDraw.Draw(canvas)
        title_font = _load_font(27, bold=True)
        body_font = _load_font(17)
        title = "暂无好感度记录"
        body = "和 AI 聊起天后，这里会出现大家的好感度。"
        title_width = self._text_width(draw, title, title_font)
        body_width = self._text_width(draw, body, body_font)
        draw.text(
            ((width - title_width) // 2, 151),
            title,
            font=title_font,
            fill=theme.text_primary,
        )
        draw.text(
            ((width - body_width) // 2, 194),
            body,
            font=body_font,
            fill=theme.text_secondary,
        )
        self._draw_footer(canvas, width, 260, accent)
        return self._save_img(canvas)
