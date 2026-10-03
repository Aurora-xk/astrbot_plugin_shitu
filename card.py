"""识别结果卡片渲染（纯 Pillow 实现，无新增依赖）。

将 AnimeTrace 识别结果（角色裁剪图 + 匹配列表）绘制成一张图片卡片，
用于替代纯文本结果发送。浅色主题：渐变页面背景 + 白色圆角卡片 +
蓝紫渐变标题横幅。
"""

import os
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

# ---- 浅色卡片配色 ----
PAGE_TOP = (243, 246, 252)
PAGE_BOTTOM = (231, 237, 246)
SURFACE = (255, 255, 255)
SURFACE_BORDER = (223, 229, 240)
BANNER_LEFT = (110, 148, 248)
BANNER_RIGHT = (162, 122, 247)
TEXT = (44, 49, 60)
MUTED = (131, 139, 153)
DIVIDER = (222, 228, 238)
THUMB_BORDER = (228, 233, 241)
RANK_COLORS = [
    (99, 141, 246),
    (76, 175, 124),
    (222, 148, 38),
    (167, 122, 245),
    (63, 173, 168),
]

CARD_WIDTH = 900
THUMB_MAX_W = 230
THUMB_MAX_H = 270

_FONT_CANDIDATES_REGULAR = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]
_FONT_CANDIDATES_BOLD = [
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]

_font_cache = {}


def _load_font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]

    font = None
    for path in _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    _font_cache[key] = font
    return font


def _truncate(draw: ImageDraw.ImageDraw, text: str, font, max_width: float) -> str:
    if not text:
        return ""
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]
    return text + "…"


def _fit_thumbnail(img: Image.Image) -> Image.Image:
    w, h = img.size
    scale = min(THUMB_MAX_W / w, THUMB_MAX_H / h, 1.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.convert("RGB").resize(new_size, Image.LANCZOS)


def _rounded(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, img.size[0] - 1, img.size[1] - 1], radius=radius, fill=255
    )
    out = img.convert("RGBA")
    out.putalpha(mask)
    return out


def _vertical_gradient(size, top, bottom) -> Image.Image:
    w, h = size
    column = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / max(1, h - 1)
        column.putpixel(
            (0, y),
            tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)),
        )
    return column.resize(size)


def _horizontal_gradient(size, left, right) -> Image.Image:
    w, h = size
    row = Image.new("RGB", (w, 1))
    for x in range(w):
        t = x / max(1, w - 1)
        row.putpixel(
            (x, 0),
            tuple(int(left[i] + (right[i] - left[i]) * t) for i in range(3)),
        )
    return row.resize(size)


def _rounded_mask(size, radius) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255
    )
    return mask


def _similarity_text(char: dict) -> str:
    raw = char.get("similarity", char.get("score"))
    if raw is None:
        return ""
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return ""
    if 0 < val <= 1.0:
        val *= 100
    if float(val).is_integer():
        return f"{int(val)}%"
    return f"{val:.1f}%"


def render_card(model_name: str, items: list, max_characters: int = 5) -> Image.Image:
    """渲染识别结果卡片。

    @param model_name: 当前识别模型显示名
    @param items: [{"index": int, "characters": [...], "crop_path": str}, ...]
    @param max_characters: 每个角色最多展示的匹配数（<=0 表示不限制）
    """
    if not items:
        raise ValueError("没有可渲染的识别结果")

    width = CARD_WIDTH
    padding = 30
    block_inner = 22
    block_gap = 18
    line_pitch = 44
    title_h = 30
    banner_h = 72
    footer_h = 104

    f_title = _load_font(34, bold=True)
    f_model = _load_font(22)
    f_block_title = _load_font(21)
    f_name = _load_font(27, bold=True)
    f_work = _load_font(23)
    f_rank = _load_font(20, bold=True)
    f_sim = _load_font(21, bold=True)
    f_foot = _load_font(19)

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    # ---- 预处理：加载裁剪图、生成文本行并测量各块高度 ----
    blocks = []
    for it in items:
        crop = None
        crop_path = it.get("crop_path")
        if crop_path and os.path.isfile(crop_path):
            try:
                crop = _fit_thumbnail(Image.open(crop_path))
            except Exception:
                crop = None

        characters = it.get("characters") or []
        limit = (
            max_characters if max_characters and max_characters > 0 else len(characters)
        )
        display = characters[:limit]

        thumb_w, thumb_h = crop.size if crop else (0, 0)
        block_w = width - padding * 2
        text_left = padding + block_inner + (thumb_w + block_inner if crop else 0)
        text_right = padding + block_w - block_inner
        text_avail = text_right - text_left

        lines = []
        for i, char in enumerate(display):
            sim = _similarity_text(char)
            sim_w = probe.textlength(sim, font=f_sim) if sim else 0
            avail_name = text_avail - 34 - ((sim_w + 18) if sim else 0)
            name = _truncate(
                probe, str(char.get("character") or "未知角色"), f_name, avail_name
            )
            name_w = probe.textlength(name, font=f_name)
            work = str(char.get("work") or "")
            if work and work != "未知作品":
                work = _truncate(
                    probe, f"《{work}》", f_work, max(0, avail_name - name_w - 14)
                )
                if len(work) <= 1:  # 截断后只剩省略号时不再展示，避免出现孤立的"…"
                    work = ""
            else:
                work = ""
            lines.append(
                {
                    "rank": i,
                    "name": name,
                    "work": work,
                    "sim": sim,
                    "name_w": name_w,
                }
            )

        text_h = title_h + 16 + len(lines) * line_pitch
        if 0 < limit < len(characters):
            text_h += 36
        img_h = thumb_h + block_inner * 2 if crop else 0
        block_h = max(text_h, img_h) + block_inner * 2

        blocks.append(
            {
                "index": it.get("index", len(blocks) + 1),
                "total": len(characters),
                "show_more": 0 < limit < len(characters),
                "limit": limit,
                "crop": crop,
                "thumb_w": thumb_w,
                "thumb_h": thumb_h,
                "lines": lines,
                "block_w": block_w,
                "block_h": block_h,
                "text_left": text_left,
            }
        )

    # ---- 计算总高度 ----
    total_h = padding + banner_h + 26
    for b in blocks:
        total_h += b["block_h"] + block_gap
    total_h -= block_gap
    total_h += footer_h

    # ---- 页面渐变背景 ----
    card = _vertical_gradient((width, total_h), PAGE_TOP, PAGE_BOTTOM)
    draw = ImageDraw.Draw(card)

    # ---- 头部渐变横幅 ----
    banner_size = (width - padding * 2, banner_h)
    banner = _horizontal_gradient(banner_size, BANNER_LEFT, BANNER_RIGHT)
    card.paste(banner, (padding, padding), _rounded_mask(banner.size, 16))

    banner_cy = padding + banner_h // 2
    draw.text(
        (padding + 24, banner_cy),
        "二次元角色识别结果",
        font=f_title,
        fill=(255, 255, 255),
        anchor="lm",
    )
    model_text = _truncate(
        draw, f"模型 {model_name}", f_model, width - padding * 2 - 260
    )
    draw.text(
        (width - padding - 24, banner_cy),
        model_text,
        font=f_model,
        fill=(227, 233, 252),
        anchor="rm",
    )

    # ---- 角色块（白色圆角卡片） ----
    y = padding + banner_h + 26
    for b in blocks:
        block_x = padding
        block_y = y
        draw.rounded_rectangle(
            [block_x, block_y, block_x + b["block_w"], block_y + b["block_h"]],
            radius=18,
            fill=SURFACE,
            outline=SURFACE_BORDER,
            width=1,
        )

        if b["crop"] is not None:
            thumb = _rounded(b["crop"], 12)
            thumb_x = block_x + block_inner
            thumb_y = block_y + (b["block_h"] - b["thumb_h"]) // 2
            card.paste(thumb, (thumb_x, thumb_y), thumb)
            draw.rounded_rectangle(
                [thumb_x - 1, thumb_y - 1, thumb_x + b["thumb_w"], thumb_y + b["thumb_h"]],
                radius=13,
                outline=THUMB_BORDER,
                width=1,
            )

        text_left = b["text_left"]
        title_cy = block_y + block_inner + title_h // 2
        draw.text(
            (text_left, title_cy),
            f"第 {b['index']} 个角色",
            font=f_block_title,
            fill=MUTED,
            anchor="lm",
        )

        line_cy = title_cy + title_h // 2 + 16 + line_pitch // 2
        if not b["lines"]:
            draw.text(
                (text_left, line_cy),
                "未识别到具体角色",
                font=f_name,
                fill=MUTED,
                anchor="lm",
            )
        for line in b["lines"]:
            rank_color = RANK_COLORS[line["rank"] % len(RANK_COLORS)]
            circle_x = text_left + 13
            draw.ellipse(
                [circle_x - 13, line_cy - 13, circle_x + 13, line_cy + 13],
                fill=rank_color,
            )
            draw.text(
                (circle_x, line_cy + 1),
                str(line["rank"] + 1),
                font=f_rank,
                fill=(255, 255, 255),
                anchor="mm",
            )

            name_x = text_left + 34
            draw.text((name_x, line_cy), line["name"], font=f_name, fill=TEXT, anchor="lm")
            if line["work"]:
                draw.text(
                    (name_x + line["name_w"] + 14, line_cy - 1),
                    line["work"],
                    font=f_work,
                    fill=MUTED,
                    anchor="lm",
                )
            if line["sim"]:
                draw.text(
                    (padding + b["block_w"] - block_inner, line_cy),
                    line["sim"],
                    font=f_sim,
                    fill=rank_color,
                    anchor="rm",
                )
            line_cy += line_pitch

        if b["show_more"]:
            draw.text(
                (text_left, line_cy - line_pitch + 36),
                f"共 {b['total']} 个结果，显示前 {b['limit']} 项",
                font=f_block_title,
                fill=MUTED,
                anchor="lm",
            )

        y += b["block_h"] + block_gap

    # ---- 底部 ----
    footer_div_y = y - block_gap + 18
    draw.line(
        [(padding, footer_div_y), (width - padding, footer_div_y)],
        fill=DIVIDER,
        width=1,
    )
    footer_cy = footer_div_y + 30
    draw.text(
        (padding, footer_cy),
        "数据来源: AnimeTrace · 仅供参考",
        font=f_foot,
        fill=MUTED,
        anchor="lm",
    )
    draw.text(
        (width - padding, footer_cy),
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        font=f_foot,
        fill=MUTED,
        anchor="rm",
    )

    return card
