import asyncio
import copy
import hashlib
import json
from collections import OrderedDict
import os
import re
import tempfile

import aiohttp

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image as MsgImage
from astrbot.api.message_components import Reply
from astrbot.api.star import Context, Star, register

from .card import render_card
from .crops import collect_crops
from .image_tool import ImageRecognitionBridge, result_json
from .reverse_search import reverse_search as upload_reverse_search

DEFAULT_CONFIG = {
    "timeout_seconds": 30,
    "prompt_send_image": "📷 请发送要识别的图片（30秒内有效）",
    "prompt_timeout": "⏰ 识别请求已超时，请重新发送命令",
    "max_characters_per_role": 5,
    "result_as_image": True,
    "no_result_reply": "🔍 未识别到图片中的角色，请更换图片后重试",
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
        self._image_bridge = ImageRecognitionBridge()
        self._reverse_lock = asyncio.Lock()
        self._reverse_cache = OrderedDict()
        self._reverse_next_request = 0
        self._reverse_closed = False

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
        async with self._session.get(self.model_list_url) as response:
            response.raise_for_status()
            result = await response.json()

        self._models = result.get("data", [])
        enabled_models = [m for m in self._models if m.get("enabled", False)]
        self._default_model = next(
            (m for m in enabled_models if m.get("default", False)),
            enabled_models[0] if enabled_models else None,
        )
        self._model_cache_time = asyncio.get_event_loop().time()
        logger.debug(f"已加载模型列表: {[m['name'] for m in self._models]}")

    async def _get_default_model(self) -> dict:
        current_time = asyncio.get_event_loop().time()
        if current_time - self._model_cache_time > self._model_cache_ttl:
            await self._fetch_models()

        return self._current_model or self._default_model

    @filter.llm_tool(name="identify_anime_character")
    async def identify_anime_character(self, event: AstrMessageEvent, image_index: int = 1) -> str:
        """通过 AnimeTrace 实际检索用户图片中的动漫、GalGame 或二次元游戏角色候选。

        用户询问图片中人物是谁、角色姓名或出处时使用；不是通用看图描述工具。
        只读取本条消息图片、明确引用的图片，或同一发送者在同一会话最近 5 分钟发的图片。
        不接受 URL/路径。可能未收录新角色、真人、截图或画风变化，候选不等于确认。
        返回 not_confident=true/缺失、无命中或错误时，此接口无法确认；可继续调用
        reverse_image_search 上传原图查插画索引的角色/作品标注，不要就此结束或猜名字。
        可进一步读取返回来源核实；候选与高相似度匹配均不等于独立身份确认。

        Args:
            image_index(int): 图片序号，从 1 开始，最多 4；默认 1。
        """
        if self._session is None:
            return result_json({"status": "error", "error_code": "plugin_not_initialized"})
        try:
            default_model = await self._get_default_model()
        except Exception:
            default_model = None
        if not default_model or not default_model.get("id"):
            return result_json({"status": "error", "error_code": "recognition_models_unavailable",
                                "guidance": "本次无法识图，不要猜测人物姓名。"})
        return result_json(await self._image_bridge.recognize(
            event, default_model["id"], self._session,
            self._call_bounded_image_api, image_index))


    async def _call_bounded_image_api(self, jpeg: bytes, model: str) -> dict:
        """One bounded multipart request; signed chat image URLs never leave the host."""
        form = aiohttp.FormData()
        form.add_field("is_multi", "1")
        form.add_field("model", model)
        form.add_field("ai_detect", "0")
        form.add_field("file", jpeg, filename="image.jpg", content_type="image/jpeg")
        async with self._session.post(self.api_url, data=form, allow_redirects=False,
                                      timeout=aiohttp.ClientTimeout(total=20)) as response:
            if response.status != 200:
                raise ValueError("recognition_service_unavailable")
            chunks = []
            size = 0
            async for chunk in response.content.iter_chunked(32 * 1024):
                size += len(chunk)
                if size > 512 * 1024:
                    raise ValueError("recognition_service_unavailable")
                chunks.append(chunk)
            payload = json.loads(b"".join(chunks))
            if not isinstance(payload, dict) or payload.get("code") not in (0, 17720, 200, 17721):
                raise ValueError("recognition_service_unavailable")
            return payload


    @filter.on_llm_request()
    async def image_identity_guidance(self, event: AstrMessageEvent, req):
        # Selection metadata only: never download, get_msg, or call an API here.
        self._image_bridge.remember(event)
        has_image = any(isinstance(m, (MsgImage, Reply)) for m in event.get_messages())
        has_recent = self._image_bridge._key(event) in self._image_bridge.recent
        marker = "[AnimeTrace 图片身份核实]"
        if (has_image or has_recent) and marker not in (req.system_prompt or ""):
            req.system_prompt = (req.system_prompt or "") + (
                "\n" + marker + "用户要求上传图片以图搜图、查图片出处/原图/作者或相似图时，"
                "调用 reverse_image_search，将本轮或引用图片上传检索；不得用文字搜索冒充以图搜图。"
                "上传检索后先回答可能是什么，再给作品、作者、匹配分数与出处；"
                "必须读取返回的 characters/material/creator 标注，不能只报告引擎名或丢几个链接。"
                "默认不发相似图片；只有 Pinterest 等相似页面不算识别完成，需继续核实。"
                "用户仅询问二次元人物身份时可先调用 identify_anime_character；"
                "普通图片描述、数学题无需调用。读取返回来源核实，文字资料不等于原图反查。"
                "AnimeTrace not_confident=true/缺失、无命中或故障时，"
                "继续用 reverse_image_search 查原图，而不是猜名字或立即结束。"
                "相互独立的高相似度索引标注可支持身份候选，但不能声称最早出处。"
                "明确说明证据局限；不要声称联网便一定能识别。"
            )


    @filter.llm_tool(name="reverse_image_search")
    async def reverse_image_search(self, event: AstrMessageEvent,
                                   image_index: int = 1, max_results: int = 5) -> str:
        """上传用户图片联网检索，优先返回供文字识别和解释使用的标题、摘要及出处候选。

        普通成员可用。用户说“以图搜图”“上传这张图搜索”“找原图/出处/作者”时用此工具。
        实际上传当前消息、明确引用，或同发送者同会话最近5分钟的图片；不是按关键词下载图片。
        不接受模型指定的URL/文件路径。优先SauceNAO上传检索，匹配不足再查Yandex相似页面。
        无Key、无收费回退；结果含来源链接和搜索结果页。相似不等于最早出处，低相似度不确认身份。
        用户最需要文字结果：先回答图中可能是什么；characters是角色、material是作品、creator是作者，
        必须结合这些索引标注与similarity给出身份候选及依据，需要时读取来源网页。
        只有相似页面且recognition_status=unconfirmed时不算完成识别，应继续核实或准确说明仍缺什么。
        默认不发送相似图片，只给必要的文字和出处链接；图片只在用户明确要求时附上。
        没找到就明确说明，不用大模型猜名字、文字搜索或生成图片冒充上传搜图结果。

        Args:
            image_index(int): 本条或引用消息的图片序号，1到4，默认1。
            max_results(int): 最多返回的出处候选数量，1到8，默认5。
        """
        if (type(image_index) is not int or not 1 <= image_index <= 4
                or type(max_results) is not int or not 1 <= max_results <= 8):
            return result_json({"status": "error", "error_code": "invalid_reverse_image_parameters"})
        async with self._reverse_lock:
            if self._reverse_closed or self._session is None:
                return result_json({"status": "error", "error_code": "plugin_unavailable"})
            images, origin = await self._image_bridge.select(event)
            if len(images) < image_index:
                return result_json({"status": "no_image", "guidance": "请让用户上传或明确引用要以图搜图的图片。"})
            try:
                async with self._image_bridge.lock:
                    raw = await self._image_bridge._image_bytes(images[image_index - 1], self._session)
                    key = (hashlib.sha256(raw).hexdigest(), max_results)
                    now = asyncio.get_running_loop().time()
                    for old, row in list(self._reverse_cache.items()):
                        if row[0] <= now: del self._reverse_cache[old]
                    cached = self._reverse_cache.get(key)
                    if cached:
                        result = copy.deepcopy(cached[1])
                        result.update(
                            cached=True, image_source=origin, image_index=image_index
                        )
                        return result_json(result)
                    if now < self._reverse_next_request:
                        return result_json({"status": "cooldown", "guidance": "上传搜图冷却中，请勿本轮立即重复调用。"})
                    jpeg = await self._image_bridge._prepare_with_cancellation(raw)
                self._reverse_next_request = asyncio.get_running_loop().time() + 10
                result = await asyncio.wait_for(upload_reverse_search(jpeg, max_results), timeout=65)
            except asyncio.CancelledError:
                raise
            except Exception:
                return result_json({"status": "error", "error_code": "reverse_image_search_failed",
                                    "guidance": "图片读取、上传或搜图服务失败，不要声称已找到出处。"})
            if self._reverse_closed:
                return result_json({"status": "error", "error_code": "plugin_unavailable"})
            result.update(image_source=origin, image_index=image_index)
            self._reverse_cache[key] = (asyncio.get_running_loop().time() + (900 if result.get("status") == "ok" else 60), copy.deepcopy(result))
            while len(self._reverse_cache) > 64: self._reverse_cache.popitem(last=False)
            result["cached"] = False
            return result_json(result)


    @filter.command("以图搜图")
    async def reverse_image_command(self, event: AstrMessageEvent):
        result = json.loads(await self.reverse_image_search(event))
        lines = ["以图搜图：" + result.get("engine", "")]
        for item in result.get("matches", []):
            score = item.get("similarity")
            for key, label in (("characters", "角色标注"), ("material", "作品标注"), ("creator", "作者标注")):
                if item.get(key): lines.append(label + "：" + item[key])
            lines.append(f"{item.get('title', '出处候选')}" + (f"（相似度 {score}%）" if score is not None else "") + "\n" + item["url"])
        if not result.get("matches"):
            lines.append("未找到可信候选。" if result.get("status") == "no_match" else result.get("guidance", "请附上或引用图片后再试。"))
        if result.get("search_url"): lines.append("搜图结果页：" + result["search_url"])
        lines.append("以上是匹配图片的索引标注，相似图不一定是原作者或最早出处。")
        await event.send(event.plain_result("\n\n".join(lines)))


    @filter.command("识别")
    async def trace_search(self, event: AstrMessageEvent):
        default_model = await self._get_default_model()
        return await self.handle_image_recognition(event, default_model["id"])

    @filter.command("头像识别")
    async def avatar_trace_search(self, event: AstrMessageEvent):
        default_model = await self._get_default_model()
        return await self.handle_avatar_recognition(event, default_model["id"])

    @filter.command("amt model")
    async def model_list(self, event: AstrMessageEvent, args=None):
        await self._fetch_models()

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
        self._image_bridge.remember(event)
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
            elif os.path.isfile(image_url):
                results = await self.call_animetrace_api_with_file(image_url, model)
            else:
                raise Exception("不支持的图片来源")

            await self.send_combined_result(event, image_url, results, model)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"识别失败: {error_msg}")

            if isinstance(e, asyncio.TimeoutError):
                user_msg = "❌ 识别超时，请稍后重试"
            elif "HTTP 500" in error_msg:
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

    def _get_model_name(self, model_id: str) -> str:
        """根据模型ID获取显示名称"""
        model = next((m for m in self._models if m["id"] == model_id), None)
        if model:
            return model.get("name", model_id)
        return model_id

    @staticmethod
    async def _parse_api_response(response) -> dict:
        """解析API响应；非JSON响应时抛出带HTTP状态码的异常。"""
        try:
            return await response.json()
        except Exception:
            raise Exception(f"API错误: HTTP {response.status}")

    @staticmethod
    def _check_api_code(result: dict):
        """校验API返回码，非成功码时抛出带错误信息的异常。"""
        code = result.get("code")
        if code in (0, 17720, 200, 17721):
            return
        error_msg = result.get("zh_message") or API_ERROR_CODES.get(
            code, f"未知错误 (code={code})"
        )
        logger.warning(f"API返回错误码: {code}, 消息: {error_msg}")
        raise Exception(f"API错误: {error_msg}")

    async def call_animetrace_api_with_file(self, file_path: str, model: str) -> dict:
        model_name = self._get_model_name(model)
        logger.debug(f"调用API - 模型: {model_name} (file方式)")

        with open(file_path, "rb") as f:
            file_data = f.read()

        form = aiohttp.FormData()
        form.add_field("is_multi", "1")
        form.add_field("model", model)
        form.add_field("ai_detect", "0")
        form.add_field("file", file_data, filename="image.jpg", content_type="image/jpeg")

        async with self._session.post(self.api_url, data=form) as response:
            result = await self._parse_api_response(response)

        self._check_api_code(result)
        logger.debug(f"API返回: {len(result.get('data', []))} 个结果")
        return result

    async def call_animetrace_api_with_url(self, image_url: str, model: str) -> dict:
        payload = {"url": image_url, "is_multi": 1, "model": model, "ai_detect": 0}
        model_name = self._get_model_name(model)
        logger.debug(f"调用API - 模型: {model_name} (URL方式)")

        async with self._session.post(self.api_url, data=payload) as response:
            result = await self._parse_api_response(response)

        self._check_api_code(result)
        logger.debug(f"API返回: {len(result.get('data', []))} 个结果")
        return result

    @staticmethod
    def _format_match_line(i: int, char: dict) -> str:
        """格式化单条匹配结果；未知作品不显示。"""
        name = char.get("character", "未知角色")
        work = char.get("work") or ""
        if work and work != "未知作品":
            return f"{i + 1}. {name} - 《{work}》"
        return f"{i + 1}. {name}"

    def format_results(self, data: dict, model: str) -> str:
        results = [item for item in data["data"] if item.get("character")]

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
                lines.append(self._format_match_line(i, char))

            if limit > 0 and len(characters) > limit:
                lines.append(f"共 {len(characters)} 个结果，显示前{limit}项")

        lines.append("数据来源: AnimeTrace，仅供参考")
        lines.append(f"当前模型: {model_name}")

        return "\n".join(lines)

    async def _render_result_card(self, model: str, crop_items: list) -> str:
        """将识别结果渲染为图片卡片，返回图片文件路径。"""
        model_name = self._get_model_name(model)
        card_img = render_card(
            model_name,
            crop_items,
            max_characters=self.max_characters_per_role,
        )
        tmp_dir = tempfile.mkdtemp(prefix="astrbot_shitu_card_")
        out_path = os.path.join(tmp_dir, "result_card.png")
        card_img.save(out_path, format="PNG")
        logger.debug(f"识别结果已渲染为卡片: {out_path}")
        return out_path

    async def send_combined_result(
        self, event: AstrMessageEvent, image_url: str, results: dict, model: str
    ):
        data_list = results.get("data") or []
        if not any(item.get("character") for item in data_list):
            await event.send(event.plain_result(self.no_result_reply))
            return

        crop_items = await collect_crops(self._session, image_url, data_list)

        # 将识别结果渲染为图片卡片发送
        if self.result_as_image and crop_items:
            try:
                card_path = await self._render_result_card(model, crop_items)
                await event.send(
                    event.chain_result([Comp.Image.fromFileSystem(card_path)])
                )
                return
            except Exception as e:
                logger.warning(f"结果卡片渲染失败，回退到文字模式: {str(e)}")

        # 卡片关闭或渲染失败：发送裁剪图 + 文字
        if not crop_items:
            await event.send(event.plain_result(self.format_results(results, model)))
            return

        chain = []
        for item in crop_items:
            chain.append(Comp.Image.fromFileSystem(item["crop_path"]))

            characters = item["characters"]
            if characters:
                text_lines = []
                if len(crop_items) > 1:
                    text_lines.append(f"第 {item['index']} 个角色：")

                limit = self.max_characters_per_role
                display_characters = characters[:limit] if limit > 0 else characters
                for i, char in enumerate(display_characters):
                    text_lines.append(self._format_match_line(i, char))

                if limit > 0 and len(characters) > limit:
                    text_lines.append(f"共 {len(characters)} 个结果，显示前{limit}项")

                if text_lines:
                    chain.append(Comp.Plain("\n".join(text_lines)))
                    chain.append(Comp.Plain(""))

        if len(crop_items) < len(data_list):
            chain.append(Comp.Plain(self.format_results(results, model)))
        else:
            model_name = self._get_model_name(model)
            chain.append(Comp.Plain(f"💡 数据来源: AnimeTrace，仅供参考\n当前模型: {model_name}"))

        await event.send(event.chain_result(chain))

    async def timeout_check(self, user_id: str):
        await asyncio.sleep(self.timeout_seconds)
        if user_id in self.waiting_sessions:
            session = self.waiting_sessions[user_id]
            event = session["event"]
            del self.waiting_sessions[user_id]
            del self.timeout_tasks[user_id]
            await event.send(event.plain_result(self.prompt_timeout))
            logger.debug(f"用户 {user_id} 的图片识别请求已超时")

    async def terminate(self):
        self._reverse_closed = True
        self._reverse_cache.clear()
        self._image_bridge.clear()
        logger.info("AnimeTrace图片识别插件已卸载")
        for task in self.timeout_tasks.values():
            task.cancel()
        self.timeout_tasks.clear()
        if self._session:
            await self._session.close()
