"""
表情包管理模块 - StickerManager

管理本地表情包目录，支持按分类随机选取图片。
"""

import random
from pathlib import Path
from typing import Optional


STICKER_EXTENSIONS = (".jpg", ".png", ".gif", ".webp")


class StickerManager:
    """管理本地表情包目录，按分类文件夹组织。"""

    def __init__(self, sticker_dir: Path):
        self.sticker_dir = sticker_dir
        sticker_dir.mkdir(parents=True, exist_ok=True)

    def get_categories(self) -> list[str]:
        """返回所有分类名（子目录名）。"""
        if not self.sticker_dir.exists():
            return []
        return [d.name for d in self.sticker_dir.iterdir() if d.is_dir()]

    def get_random_sticker(self, category: str) -> Optional[Path]:
        """从指定分类随机返回一张表情包的绝对路径，分类不存在或为空时返回 None。"""
        cat_path = self.sticker_dir / category
        if not cat_path.exists() or not cat_path.is_dir():
            return None
        files = [
            f
            for f in cat_path.iterdir()
            if f.is_file() and f.suffix.lower() in STICKER_EXTENSIONS
        ]
        if not files:
            return None
        return random.choice(files).resolve()
