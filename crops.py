"""角色裁剪：下载原图并按识别框裁剪角色区域，供卡片和文字结果共用。"""

import os
import tempfile
import urllib.parse
from io import BytesIO

from PIL import Image as PILImage

from astrbot.api import logger

# 单次识别最多裁剪的角色数，防止卡片过长
MAX_CROPS = 5


async def download_image_data(session, image_url: str) -> bytes:
    """下载图片数据（支持本地路径、file:// URI和HTTP/HTTPS URL）"""
    if os.path.isfile(image_url):
        with open(image_url, "rb") as f:
            return f.read()

    if image_url.startswith("file://"):
        file_path = urllib.parse.unquote(image_url.replace("file://", ""))
        if os.name == "nt" and file_path.startswith("/"):
            file_path = file_path[1:]
        with open(file_path, "rb") as f:
            return f.read()

    if image_url.startswith("telegram://"):
        raise Exception("Telegram文件暂不支持")

    async with session.get(image_url) as response:
        if response.status != 200:
            raise Exception(f"图片下载失败: HTTP {response.status}")
        return await response.read()


def make_crops(img: PILImage.Image, data_list: list) -> list:
    """按识别框裁剪角色区域，返回卡片/文字结果共用的数据结构。

    每项: {"index": 原始序号, "characters": [...], "crop_path": str}
    """
    tmp_dir = tempfile.mkdtemp(prefix="astrbot_shitu_crops_")
    w, h = img.size
    items = []

    for idx, item in enumerate(data_list, start=1):
        if len(items) >= MAX_CROPS:
            break

        box = item.get("box")
        if not box or len(box) != 4:
            continue

        x1 = int(max(0, min(1, float(box[0]))) * w)
        y1 = int(max(0, min(1, float(box[1]))) * h)
        x2 = int(max(0, min(1, float(box[2]))) * w)
        y2 = int(max(0, min(1, float(box[3]))) * h)

        if x2 <= x1 or y2 <= y1:
            continue

        cropped = img.crop((x1, y1, x2, y2))
        out_path = os.path.join(tmp_dir, f"crop_{idx}.jpg")
        cropped.save(out_path, format="JPEG", quality=90)
        items.append(
            {
                "index": idx,
                "characters": item.get("character") or [],
                "crop_path": out_path,
            }
        )

    return items


async def collect_crops(session, image_url: str, data_list: list) -> list:
    """下载原图并裁剪角色区域；失败时返回空列表（由调用方回退到纯文本）。"""
    try:
        img_data = await download_image_data(session, image_url)
        img = PILImage.open(BytesIO(img_data)).convert("RGB")
    except Exception as e:
        logger.debug(f"原图下载失败，跳过角色裁剪: {str(e)}")
        return []

    return make_crops(img, data_list)
