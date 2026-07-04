import asyncio
import os
import re
import tempfile
import urllib.parse
from io import BytesIO

import aiohttp
from PIL import Image as PILImage

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image as MsgImage
from astrbot.api.message_components import Reply
from astrbot.api.star import Context, Star, register

DEFAULT_CONFIG = {
    "timeout_seconds": 30,
    "prompt_send_image": "📷 请发送要识别的图片（30秒内有效）",
    "prompt_timeout": "⏰ 识别请求已超时，请重新发送命令",
    "return_crops": True,
    "max_crops": 5,
    "max_characters_per_role": 5,
    "forward_threshold": 0,
}

API_ERROR_CODES = {
    17720: "识别成功",
    200: "Success",
    17721: "服务器正常运行中",
    17701: "图片大小过大",
    17702: "服务器繁忙，请重试",
    17703: "请求参数不正确",
    17704: "API维护中",
    17705: "图片格式不支持",
    17706: "识别无法完成（内部错误，请重试）",
    17707: "内部错误",
    17708: "图片中的人物数量超过限制",
    17722: "图片下载失败",
    17728: "已达到本次使用上限",
    17731: "服务利用人数过多，请重新尝试",
    404: "页面不存在",
}


@register(
    "astrbot_plugin_shitu",
    "aurora",
    "AnimeTrace图片识别插件",
    "4.0",
    "https://github.com/Aurora-xk/astrbot_plugin_shitu",
)
class AnimeTracePlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.api_url: str = "https://api.animetrace.com/v1/search"
        self.model_list_url: str = "https://api.animetrace.com/v1/model/list"
        self.waiting_sessions = {}
        self.timeout_tasks = {}
        self._session = None
        self._models = []
        self._default_model = None
        self._current_model = None
        self._model_cache_time = 0
        self._model_cache_ttl = 3600

        shitu_config = (
            config.get("shitu_settings", {})
            if config
            else getattr(self.context, "_config", {}).get("shitu_settings", {})
        )
        for key, default in DEFAULT_CONFIG.items():
            setattr(self, key, shitu_config.get(key, default))

    async def initialize(self):
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        await self._fetch_models()
        logger.info("AnimeTrace图片识别插件已加载")

    async def _fetch_models(self):
        try:
            async with self._session.get(self.model_list_url) as response:
                if response.status != 200:
                    logger.warning(f"获取模型列表失败: HTTP {response.status}")
                    return

                result = await response.json()
                if result.get("code") != 0:
                    logger.warning(
                        f"获取模型列表失败: {result.get('message', '未知错误')}"
                    )
                    return

                self._models = result.get("data", [])
                enabled_models = [m for m in self._models if m.get("enabled", False)]
                self._default_model = next(
                    (m for m in enabled_models if m.get("default", False)),
                    enabled_models[0] if enabled_models else None,
                )
                self._model_cache_time = asyncio.get_event_loop().time()

                model_names = [m["name"] for m in self._models]
                logger.debug(f"已加载模型列表: {model_names}")
        except Exception as e:
            logger.warning(f"获取模型列表异常: {str(e)}")

    async def _get_default_model(self) -> dict:
        current_time = asyncio.get_event_loop().time()
        if (
            not self._models
            or current_time - self._model_cache_time > self._model_cache_ttl
        ):
            await self._fetch_models()

        if self._current_model:
            return self._current_model

        return self._default_model or self._models[0]

    @filter.command("识别")
    async def trace_search(self, event: AstrMessageEvent, args=None):
        default_model = await self._get_default_model()
        return await self.handle_image_recognition(event, default_model["id"])

    @filter.command("头像识别")
    async def avatar_trace_search(self, event: AstrMessageEvent, args=None):
        default_model = await self._get_default_model()
        return await self.handle_avatar_recognition(event, default_model["id"])

    @filter.command("amt model")
    async def model_list(self, event: AstrMessageEvent, args=None):
        await self._fetch_models()

        if not self._models:
            await event.send(event.plain_result("❌ 无法获取模型列表，请稍后重试"))
            return

        if args is not None:
            try:
                index = int(args) - 1
            except (ValueError, TypeError):
                await event.send(event.plain_result("❌ 无效的模型编号"))
                return
            if 0 <= index < len(self._models):
                model = self._models[index]
                if model.get("enabled", False):
                    self._current_model = model
                    await event.send(
                        event.plain_result(
                            f"✅ 已切换到模型: {model['id']}"
                        )
                    )
                else:
                    await event.send(
                        event.plain_result(
                            f"❌ 模型 {model['id']} 当前不可用，请选择其他模型"
                        )
                    )
            else:
                await event.send(event.plain_result("❌ 无效的模型编号"))
            return

        lines = ["📋 AnimeTrace 模型列表："]
        current_model_id = self._current_model["id"] if self._current_model else None
        for idx, model in enumerate(self._models, start=1):
            model_id = model["id"]
            desc = model.get("desc", {})
            desc_zh = desc.get("zh", "")
            enabled = model.get("enabled", True)
            is_current = model_id == current_model_id

            line = f"{idx}. {model_id}"
            if is_current:
                line += " ⭐(当前)"
            if desc_zh:
                line += f"\n   {desc_zh}"
            line += f"\n   状态: {'✅ 可用' if enabled else '❌ 不可用'}"
            lines.append(line)

        lines.append("\n使用 /amt model 数字 切换模型")
        await event.send(event.plain_result("\n".join(lines)))

    async def handle_image_recognition(self, event: AstrMessageEvent, model: str):
        user_id = event.get_sender_id()

        image_url = await self.extract_image_from_event(event)
        if image_url:
            await self.process_image_recognition(event, image_url, model)
            return

        try:
            raw_event = event._event if hasattr(event, "_event") else event
            if hasattr(raw_event, "reply_to_message") and raw_event.reply_to_message:
                logger.debug("检测到引用消息，但引用消息中没有找到图片")
                await event.send(
                    event.plain_result(
                        "❌ 引用消息中没有找到图片，请确保引用的消息包含图片"
                    )
                )
                return
        except Exception as e:
            logger.warning(f"检查引用消息状态时出错: {str(e)}")

        self.waiting_sessions[user_id] = {
            "model": model,
            "timestamp": asyncio.get_event_loop().time(),
            "event": event,
        }

        if user_id in self.timeout_tasks:
            self.timeout_tasks[user_id].cancel()

        timeout_task = asyncio.create_task(self.timeout_check(user_id))
        self.timeout_tasks[user_id] = timeout_task

        await event.send(event.plain_result(self.prompt_send_image))
        logger.debug(f"用户 {user_id} 进入等待图片状态，等待{self.timeout_seconds}秒")

    async def handle_avatar_recognition(self, event: AstrMessageEvent, model: str):
        try:
            mentioned_user_id = await self.extract_mentioned_user(event)

            if not mentioned_user_id:
                mentioned_user_id = event.get_sender_id()
                await event.send(event.plain_result("📸 识别您自己的头像..."))
            else:
                full_text = self._get_full_text(event.get_messages())
                qq_match = re.search(r"头像识别\s*(\d{5,12})", full_text)
                if qq_match and qq_match.group(1) == mentioned_user_id:
                    await event.send(
                        event.plain_result(f"📸 识别QQ号 {mentioned_user_id} 的头像...")
                    )

            avatar_url = (
                f"https://q.qlogo.cn/headimg_dl?dst_uin={mentioned_user_id}&spec=640"
            )
            event._avatar_command_processed = True

            await self.process_image_recognition(event, avatar_url, model)

        except Exception as e:
            logger.error(f"头像识别失败: {str(e)}")
            await event.send(event.plain_result(f"❌ 头像识别失败: {str(e)}"))

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        user_id = event.get_sender_id()

        messages = event.get_messages()
        full_text = self._get_full_text(messages)

        if not hasattr(event, "_avatar_command_processed"):
            if re.search(r"头像识别", full_text):
                event._avatar_command_processed = True
                default_model = await self._get_default_model()
                await self.handle_avatar_recognition(event, default_model["id"])
                return

        if user_id not in self.waiting_sessions:
            return

        session = self.waiting_sessions[user_id]

        current_time = asyncio.get_event_loop().time()
        if current_time - session["timestamp"] > self.timeout_seconds:
            return

        image_url = await self.extract_image_from_event(event)
        if not image_url:
            return

        del self.waiting_sessions[user_id]
        if user_id in self.timeout_tasks:
            self.timeout_tasks[user_id].cancel()
            del self.timeout_tasks[user_id]
        await self.process_image_recognition(event, image_url, session["model"])

    async def process_image_recognition(
        self, event: AstrMessageEvent, image_url: str, model: str
    ):
        try:
            if image_url.startswith(("http://", "https://")):
                results = await self.call_animetrace_api_with_url(image_url, model)
                if not results or not results.get("data"):
                    logger.debug("URL识别方式未返回结果，尝试file方式...")
                    temp_path = await self.download_to_temp_file(image_url)
                    if temp_path:
                        results = await self.call_animetrace_api_with_file(
                            temp_path, model
                        )
            elif os.path.isfile(image_url):
                results = await self.call_animetrace_api_with_file(image_url, model)
            else:
                raise Exception("不支持的图片来源")

            await self.send_combined_result(event, image_url, results, model)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"识别失败: {error_msg}")

            if "HTTP 500" in error_msg:
                user_msg = "❌ 识别服务暂时不可用，请稍后重试"
            elif "HTTP 422" in error_msg:
                user_msg = "❌ 图片格式不支持，请尝试其他图片"
            elif "HTTP 413" in error_msg or "图片大小过大" in error_msg:
                user_msg = "❌ 图片大小过大，请使用更小的图片"
            elif "HTTP 403" in error_msg or "API维护中" in error_msg:
                user_msg = "❌ API维护中，请稍后重试"
            elif "服务器繁忙" in error_msg or "服务利用人数过多" in error_msg:
                user_msg = "❌ 服务器繁忙，请稍后重试"
            elif "达到本次使用上限" in error_msg:
                user_msg = "❌ 已达到本次使用上限"
            elif "人物数量超过限制" in error_msg:
                user_msg = "❌ 图片中的人物数量超过限制"
            elif "图片格式不支持" in error_msg:
                user_msg = "❌ 图片格式不支持，请尝试其他图片"
            elif "图片下载失败" in error_msg:
                user_msg = "❌ 图片下载失败，请重试"
            elif "timeout" in error_msg.lower():
                user_msg = "❌ 识别超时，请稍后重试"
            else:
                user_msg = f"❌ 识别失败: {error_msg}"

            try:
                await event.send(event.plain_result(user_msg))
            except Exception as send_error:
                logger.warning(f"发送错误消息失败: {send_error}")

    def _get_full_text(self, messages) -> str:
        """从消息列表中提取完整文本"""
        full_text = ""
        for msg in messages:
            if hasattr(msg, "text"):
                full_text += str(msg.text)
            elif hasattr(msg, "type") and msg.type == "Plain":
                full_text += str(msg)
        return full_text

    async def extract_mentioned_user(self, event: AstrMessageEvent) -> str:
        messages = event.get_messages()
        full_text = self._get_full_text(messages)

        qq_match = re.search(r"头像识别\s*(\d{5,12})", full_text)
        if qq_match:
            return qq_match.group(1)

        for msg in messages:
            if hasattr(msg, "type") and msg.type == "At":
                if hasattr(msg, "qq"):
                    return str(msg.qq)
                if hasattr(msg, "user_id"):
                    return str(msg.user_id)

            if hasattr(msg, "text"):
                text = str(msg.text)
                at_match = re.search(r"\[CQ:at,qq=(\d+)\]", text)
                if at_match:
                    return at_match.group(1)

        return None

    async def extract_image_from_event(self, event: AstrMessageEvent) -> str:
        messages = event.get_messages()

        for msg in messages:
            if isinstance(msg, MsgImage):
                image_ref = self._get_image_reference(msg)
                if image_ref:
                    try:
                        if hasattr(msg, "convert_to_file_path"):
                            file_path = await msg.convert_to_file_path()
                            if file_path and os.path.isfile(file_path):
                                return file_path
                    except Exception as e:
                        logger.debug(f"convert_to_file_path失败: {str(e)}")

                    if image_ref.startswith(("http://", "https://")):
                        return image_ref.strip("`'").strip()

        try:
            raw_message = getattr(event.message_obj, "raw_message", None)
            if raw_message:
                attachments = getattr(raw_message, "attachments", None)
                if attachments and isinstance(attachments, list):
                    for attachment in attachments:
                        url = getattr(attachment, "url", None)
                        if (
                            url
                            and isinstance(url, str)
                            and url.startswith(("http://", "https://"))
                        ):
                            return url.strip("`'").strip()

                item_list = (
                    raw_message.get("item_list")
                    if isinstance(raw_message, dict)
                    else getattr(raw_message, "item_list", None)
                )
                if item_list and isinstance(item_list, list):
                    for item in item_list:
                        item_type = int(item.get("type") or 0)
                        if item_type == 2:
                            image_item = item.get("image_item", {})
                            media = image_item.get("media", {})
                            encrypted_query_param = str(
                                media.get("encrypt_query_param", "")
                            ).strip()
                            if encrypted_query_param:
                                cdn_url = f"https://novac2c.cdn.weixin.qq.com/c2c/download?encrypted_query_param={encrypted_query_param}"
                                return cdn_url
        except Exception as e:
            logger.debug(f"从raw_message提取图片URL失败: {str(e)}")

        try:
            for msg in messages:
                if isinstance(msg, Reply) and hasattr(msg, "chain") and msg.chain:
                    for reply_msg in msg.chain:
                        if isinstance(reply_msg, MsgImage):
                            image_ref = self._get_image_reference(reply_msg)
                            if image_ref:
                                try:
                                    if hasattr(reply_msg, "convert_to_file_path"):
                                        file_path = (
                                            await reply_msg.convert_to_file_path()
                                        )
                                        if file_path and os.path.isfile(file_path):
                                            return file_path
                                except Exception as e:
                                    logger.debug(
                                        f"引用消息convert_to_file_path失败: {str(e)}"
                                    )

                                if image_ref.startswith(("http://", "https://")):
                                    return image_ref.strip("`'").strip()
        except Exception as e:
            logger.warning(f"检查引用消息图片时出错: {str(e)}")

        return None

    def _get_image_reference(self, msg) -> str:
        """获取图片组件的引用（优先url，其次file）"""
        return getattr(msg, "url", None) or getattr(msg, "file", None)

    async def _download_image_data(self, image_url: str) -> bytes:
        """下载图片数据（支持本地路径、file:// URI和HTTP/HTTPS URL）"""
        if os.path.isfile(image_url):
            logger.debug(f"读取本地图片: {image_url}")
            with open(image_url, "rb") as f:
                return f.read()

        if image_url.startswith("file://"):
            file_path = urllib.parse.unquote(image_url.replace("file://", ""))
            if os.name == "nt" and file_path.startswith("/"):
                file_path = file_path[1:]
            logger.debug(f"读取file://图片: {file_path}")
            with open(file_path, "rb") as f:
                return f.read()

        if image_url.startswith("telegram://"):
            raise Exception("Telegram文件暂不支持")

        async with self._session.get(image_url) as response:
            if response.status != 200:
                raise Exception(f"图片下载失败: HTTP {response.status}")
            return await response.read()

    async def download_to_temp_file(self, image_url: str) -> str:
        logger.debug(f"下载图片到临时文件: {image_url[:100]}...")

        try:
            img_data = await self._download_image_data(image_url)

            img = PILImage.open(BytesIO(img_data))

            if max(img.size) > 1024:
                ratio = 1024 / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, PILImage.LANCZOS)

            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".jpg", delete=False
            ) as f:
                img.save(f, format="JPEG", quality=85)
                temp_path = f.name

            logger.debug(f"图片保存到临时文件: {temp_path}")
            return temp_path
        except asyncio.TimeoutError:
            raise Exception("图片下载超时，请稍后重试")
        except Exception as e:
            logger.error(f"图片下载失败: {str(e)}")
            raise Exception(f"图片下载失败: {str(e)}")

    def _get_model_name(self, model_id: str) -> str:
        """根据模型ID获取显示名称"""
        model = next((m for m in self._models if m["id"] == model_id), None)
        if model:
            return model.get("name", model_id)
        return model_id

    async def call_animetrace_api_with_file(self, file_path: str, model: str) -> dict:
        model_name = self._get_model_name(model)
        logger.debug(f"调用API - 模型: {model_name} (file方式)")

        try:
            with open(file_path, "rb") as f:
                file_data = f.read()

            form = aiohttp.FormData()
            form.add_field("is_multi", "1")
            form.add_field("model", model)
            form.add_field("ai_detect", "0")
            form.add_field("file", file_data, filename="image.jpg", content_type="image/jpeg")

            async with self._session.post(self.api_url, data=form) as response:
                try:
                    result = await response.json()
                except Exception:
                    error_text = await response.text()
                    logger.warning(
                        f"API返回错误状态: HTTP {response.status}, 响应: {error_text[:200]}"
                    )
                    raise Exception(f"API错误: HTTP {response.status}")

                code = result.get("code")

                if code not in (0, 17720, 200, 17721):
                    zh_message = result.get("zh_message", "")
                    if zh_message:
                        error_msg = zh_message
                    else:
                        error_msg = API_ERROR_CODES.get(code, f"未知错误 (code={code})")
                    logger.warning(f"API返回错误码: {code}, 消息: {error_msg}")
                    raise Exception(f"API错误: {error_msg}")

                logger.debug(f"API返回: {len(result.get('data', []))} 个结果")
                return result
        except asyncio.TimeoutError:
            logger.error("API调用超时")
            raise Exception("识别服务响应超时，请稍后重试")
        except Exception as e:
            logger.error(f"file API调用失败: {str(e)}")
            raise

    async def call_animetrace_api_with_url(self, image_url: str, model: str) -> dict:
        payload = {"url": image_url, "is_multi": 1, "model": model, "ai_detect": 0}
        model_name = self._get_model_name(model)
        logger.debug(f"调用API - 模型: {model_name} (URL方式)")

        try:
            async with self._session.post(self.api_url, data=payload) as response:
                try:
                    result = await response.json()
                except Exception:
                    if response.status in [422, 500, 502, 503, 504]:
                        logger.debug(
                            f"URL识别失败 (HTTP {response.status})，准备回退到file方式"
                        )
                        return {"data": []}
                    error_text = await response.text()
                    logger.warning(
                        f"API返回错误状态: HTTP {response.status}, 响应: {error_text[:200]}"
                    )
                    raise Exception(f"API错误: HTTP {response.status}")

                code = result.get("code")

                if code not in (0, 17720, 200, 17721):
                    if code in (17701, 17705, 17708, 17722):
                        logger.debug(
                            f"URL识别失败 (code={code})，准备回退到file方式"
                        )
                        return {"data": []}
                    zh_message = result.get("zh_message", "")
                    if zh_message:
                        error_msg = zh_message
                    else:
                        error_msg = API_ERROR_CODES.get(code, f"未知错误 (code={code})")
                    logger.warning(f"API返回错误码: {code}, 消息: {error_msg}")
                    raise Exception(f"API错误: {error_msg}")

                logger.debug(f"API返回: {len(result.get('data', []))} 个结果")
                return result
        except Exception as e:
            logger.warning(f"URL方式调用失败: {str(e)}，准备回退到file方式")
            return {"data": []}

    def format_results(self, data: dict, model: str) -> str:
        if not data.get("data") or not data["data"]:
            return "🔍 未找到匹配的信息"

        results = [item for item in data["data"] if item.get("character")]
        if not results:
            return "🔍 未识别到具体角色信息"

        model_name = self._get_model_name(model)

        lines = [f"🔍 {model_name} 识别结果"]

        for idx, item in enumerate(results, start=1):
            characters = item.get("character", [])
            if not characters:
                continue

            if len(results) > 1:
                lines.append(f"\n第 {idx} 个角色：")

            limit = self.max_characters_per_role
            display_characters = characters[:limit] if limit > 0 else characters
            for i, char in enumerate(display_characters):
                name = char.get("character", "未知角色")
                work = char.get("work", "未知作品")
                lines.append(f"{i + 1}. {name} - 《{work}》")

            if limit > 0 and len(characters) > limit:
                lines.append(f"共 {len(characters)} 个结果，显示前{limit}项")

        model_name = self._get_model_name(model)
        lines.append("数据来源: AnimeTrace，仅供参考")
        lines.append(f"当前模型: {model_name}")

        return "\n".join(lines)

    async def send_combined_result(
        self, event: AstrMessageEvent, image_url: str, results: dict, model: str
    ):
        try:
            data_list = results.get("data") or []
            if not data_list:
                response = self.format_results(results, model)
                await event.send(event.plain_result(response))
                return

            chain = []

            if self.return_crops:
                try:
                    img_data = await self._download_image_data(image_url)
                except Exception as e:
                    logger.debug(f"裁剪图片下载失败: {str(e)}")
                    response_text = self.format_results(results, model)
                    await event.send(event.plain_result(response_text))
                    return

                img = PILImage.open(BytesIO(img_data)).convert("RGB")
                w, h = img.size

                tmp_dir = tempfile.mkdtemp(prefix="astrbot_shitu_crops_")
                crop_paths = []

                for idx, item in enumerate(data_list, start=1):
                    if len(crop_paths) >= self.max_crops:
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
                    crop_paths.append((idx, out_path, item))

                for idx, out_path, item in crop_paths:
                    chain.append(Comp.Image.fromFileSystem(out_path))

                    characters = item.get("character") or []
                    if characters:
                        text_lines = []
                        if len(crop_paths) > 1:
                            text_lines.append(f"第 {idx} 个角色：")

                        limit = self.max_characters_per_role
                        display_characters = (
                            characters[:limit] if limit > 0 else characters
                        )
                        for i, char in enumerate(display_characters):
                            name = char.get("character", "未知角色")
                            work = char.get("work", "未知作品")
                            text_lines.append(f"{i + 1}. {name} - 《{work}》")

                        if limit > 0 and len(characters) > limit:
                            text_lines.append(
                                f"共 {len(characters)} 个结果，显示前{limit}项"
                            )

                        if text_lines:
                            chain.append(Comp.Plain("\n".join(text_lines)))
                            chain.append(Comp.Plain(""))

            if not self.return_crops or len(crop_paths) < len(data_list):
                response_text = self.format_results(results, model)
                chain.append(Comp.Plain(response_text))
            else:
                model_name = self._get_model_name(model)
                chain.append(Comp.Plain(f"💡 数据来源: AnimeTrace，仅供参考\n当前模型: {model_name}"))

            character_count = len(
                [item for item in data_list if item.get("character")]
            )
            use_forward = (
                self.forward_threshold > 0
                and character_count >= self.forward_threshold
                and event.get_platform_name() == "aiocqhttp"
            )

            if chain:
                if use_forward:
                    sender_name = event.get_sender_name() or "AnimeTrace"
                    sender_id = event.get_sender_id() or "10000"
                    nodes = []
                    current_content = []
                    for comp in chain:
                        if isinstance(comp, Comp.Image):
                            if current_content:
                                nodes.append(
                                    Comp.Node(
                                        content=current_content,
                                        name=sender_name,
                                        uin=sender_id,
                                    )
                                )
                                current_content = []
                            current_content.append(comp)
                        elif isinstance(comp, Comp.Plain):
                            if comp.text.strip():
                                current_content.append(comp)
                        else:
                            current_content.append(comp)
                    if current_content:
                        nodes.append(
                            Comp.Node(
                                content=current_content,
                                name=sender_name,
                                uin=sender_id,
                            )
                        )
                    if nodes:
                        await event.send(event.chain_result([Comp.Nodes(nodes)]))
                else:
                    await event.send(event.chain_result(chain))
            else:
                response_text = self.format_results(results, model)
                await event.send(event.plain_result(response_text))

        except Exception as e:
            logger.warning(f"发送合并结果失败: {e}")
            try:
                response_text = self.format_results(results, model)
                await event.send(event.plain_result(response_text))
            except Exception as send_error:
                logger.warning(f"发送文字结果也失败: {send_error}")

    async def timeout_check(self, user_id: str):
        try:
            await asyncio.sleep(self.timeout_seconds)
            if user_id in self.waiting_sessions:
                session = self.waiting_sessions[user_id]
                event = session["event"]
                del self.waiting_sessions[user_id]
                del self.timeout_tasks[user_id]
                try:
                    await event.send(event.plain_result(self.prompt_timeout))
                    logger.debug(f"用户 {user_id} 的图片识别请求已超时")
                except Exception as send_error:
                    logger.warning(f"发送超时消息失败: {send_error}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"超时检查任务异常: {str(e)}")

    async def terminate(self):
        logger.info("AnimeTrace图片识别插件已卸载")
        for task in self.timeout_tasks.values():
            task.cancel()
        self.timeout_tasks.clear()
        if self._session:
            await self._session.close()
