"""
表情包管理模块 - StickerManager

管理本地表情包目录，支持按分类随机选取图片。
"""

import random
from pathlib import Path
from typing import Optional


STICKER_EXTENSIONS = (".jpg", ".png", ".gif", ".webp")


class StickerManager:
    """管理本地表情包目录，按人格子目录与分类文件夹组织。

    目录结构示例：
      stickers/
        ├── default/           (默认/通用人格表情包，或直接放在 stickers/ 根目录)
        │     └── 开心/
        └── neko/              (人格 neko 的独立表情包目录)
              ├── 撒娇/
              └── 傲娇/
    """

    def __init__(self, sticker_dir: Path):
        self.sticker_dir = sticker_dir
        sticker_dir.mkdir(parents=True, exist_ok=True)

    def resolve_persona_dir(
        self, persona_id: str = "default", dir_name: Optional[str] = None
    ) -> Path:
        """解析人格对应的表情包子目录路径。"""
        sub_name = (dir_name or "").strip() or (persona_id or "").strip() or "default"
        if sub_name == "default":
            # 如果存在 stickers/default/ 优先使用，否则使用 stickers/ 根目录
            default_sub = self.sticker_dir / "default"
            if default_sub.exists() and default_sub.is_dir():
                return default_sub
            return self.sticker_dir
        return self.sticker_dir / sub_name

    def get_categories(
        self, persona_id: str = "default", dir_name: Optional[str] = None
    ) -> list[str]:
        """返回指定人格所有分类名（子目录名），不存在时回退到默认表情包目录。"""
        p_dir = self.resolve_persona_dir(persona_id, dir_name)
        if p_dir.exists() and p_dir.is_dir():
            cats = [d.name for d in p_dir.iterdir() if d.is_dir()]
            if cats:
                return sorted(cats)

        # 回退检查默认目录
        default_dir = self.resolve_persona_dir("default")
        if default_dir.exists() and default_dir.is_dir() and default_dir != p_dir:
            return sorted([d.name for d in default_dir.iterdir() if d.is_dir()])

        return []

    def get_random_sticker(
        self,
        category: str,
        persona_id: str = "default",
        dir_name: Optional[str] = None,
    ) -> Optional[Path]:
        """从指定分类随机返回一张表情包的绝对路径。优先从人格专属目录读取，为空或不存在时回退默认。"""
        # 1. 优先从人格专属目录查找
        p_dir = self.resolve_persona_dir(persona_id, dir_name)
        cat_path = p_dir / category
        if cat_path.exists() and cat_path.is_dir():
            files = [
                f
                for f in cat_path.iterdir()
                if f.is_file() and f.suffix.lower() in STICKER_EXTENSIONS
            ]
            if files:
                return random.choice(files).resolve()

        # 2. 回退到默认目录查找
        default_dir = self.resolve_persona_dir("default")
        if default_dir != p_dir:
            default_cat = default_dir / category
            if default_cat.exists() and default_cat.is_dir():
                files = [
                    f
                    for f in default_cat.iterdir()
                    if f.is_file() and f.suffix.lower() in STICKER_EXTENSIONS
                ]
                if files:
                    return random.choice(files).resolve()

        return None
