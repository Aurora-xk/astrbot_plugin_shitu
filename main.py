from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Image as MsgImage
import aiohttp
import asyncio
import base64
from io import BytesIO
from PIL import Image as PILImage


@register("astrbot_plugin_shitu", "shenx", "动漫/Gal/二游图片识别插件", "2.2.1", "https://github.com/shenxgan")
class AnimeTracePlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.api_url = "https://api.animetrace.com/v1/search"
        self.waiting_sessions = {}  # 简单的会话管理
        self.timeout_tasks = {}  # 存储超时任务

    async def initialize(self):
        logger.info("动漫/Gal/二游识别插件已加载")

    @filter.command("动漫识别", "动漫图片识别")
    async def anime_search(self, event: AstrMessageEvent, args=None):
        """使用pre_stable模型进行动漫图片识别"""
        return await self.handle_image_recognition(event, "pre_stable")

    @filter.command("gal识别", "GalGame图片识别")
    async def gal_search(self, event: AstrMessageEvent, args=None):
        """使用full_game_model_kira模型进行GalGame图片识别"""
        return await self.handle_image_recognition(event, "full_game_model_kira")

    @filter.command("通用识别", "动漫/Gal/二游图片识别")
    async def trace_search(self, event: AstrMessageEvent, args=None):
        """使用animetrace_high_beta模型进行通用图片识别"""
        return await self.handle_image_recognition(event, "animetrace_high_beta")

    async def handle_image_recognition(self, event: AstrMessageEvent, model: str):
        """简化的图片识别处理"""
        user_id = event.get_sender_id()

        # 检查当前消息是否包含图片
        image_url = await self.extract_image_from_event(event)
        if image_url:
            await self.process_image_recognition(event, image_url, model)
            return

        # 如果没有图片，设置等待状态
        self.waiting_sessions[user_id] = {
            "model": model,
            "timestamp": asyncio.get_event_loop().time(),
            "event": event,  # 保存事件对象用于超时消息发送
        }

        # 创建30秒超时任务
        if user_id in self.timeout_tasks:
            self.timeout_tasks[user_id].cancel()  # 取消之前的超时任务
        
        timeout_task = asyncio.create_task(self.timeout_check(user_id))
        self.timeout_tasks[user_id] = timeout_task

        await event.send(event.plain_result("📷 请发送要识别的图片（30秒内有效）"))
        logger.info(f"用户 {user_id} 进入等待图片状态，等待30秒")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听所有消息，处理等待中的图片识别请求"""
        user_id = event.get_sender_id()

        # 检查用户是否在等待图片识别
        if user_id not in self.waiting_sessions:
            return

        session = self.waiting_sessions[user_id]

        # 检查是否超时（30秒）
        current_time = asyncio.get_event_loop().time()
        if current_time - session["timestamp"] > 30:
            return  # 超时检查由定时任务处理，这里直接返回

        # 提取图片
        image_url = await self.extract_image_from_event(event)
        if not image_url:
            return  # 不是图片消息，继续等待

        # 找到图片，开始识别
        del self.waiting_sessions[user_id]  # 清除等待状态
        if user_id in self.timeout_tasks:
            self.timeout_tasks[user_id].cancel()  # 取消超时任务
            del self.timeout_tasks[user_id]
        await self.process_image_recognition(event, image_url, session["model"])

    async def process_image_recognition(
        self, event: AstrMessageEvent, image_url: str, model: str
    ):
        """处理图片识别"""
        try:
            # 首先尝试直接使用URL调用API（更高效）
            results = await self.call_animetrace_api_with_url(image_url, model)
            
            # 如果URL方式失败，再回退到下载图片方式
            if not results or not results.get("data"):
                logger.info("URL识别方式未返回结果，尝试下载图片识别...")
                img_data = await self.download_and_process_image(image_url)
                results = await self.call_animetrace_api(img_data, model)

            # 格式化并发送结果
            response = self.format_results(results, model)
            try:
                await event.send(event.plain_result(response))
            except Exception as send_error:
                logger.warning(f"发送识别结果失败: {send_error}")
                # 如果发送失败，记录日志但不抛出异常

        except Exception as e:
            logger.error(f"识别失败: {str(e)}")
            try:
                await event.send(event.plain_result(f"❌ 识别失败: {str(e)}"))
            except Exception as send_error:
                logger.warning(f"发送错误消息失败: {send_error}")
                # 如果错误消息也发送失败，记录日志但不抛出异常

    async def extract_image_from_event(self, event: AstrMessageEvent) -> str:
        """从事件中提取图片URL"""
        messages = event.get_messages()

        for msg in messages:
            # 标准图片组件
            if isinstance(msg, MsgImage):
                if hasattr(msg, "url") and msg.url:
                    return msg.url.strip()
                if hasattr(msg, "file") and msg.file:
                    # 从file字段提取URL - 处理微信格式
                    file_content = str(msg.file)
                    if "http" in file_content:
                        import re

                        # 提取URL并移除反引号
                        urls = re.findall(r"https?://[^\s\`\']+", file_content)
                        if urls:
                            return urls[0].strip("`'")

            # QQ官方平台特殊处理
            if hasattr(msg, "type") and msg.type == "Plain":
                text = str(msg.text) if hasattr(msg, "text") else str(msg)
                if "attachmentType=" in text and "image" in text:
                    # 这是QQ官方的图片消息格式，需要后续消息处理
                    continue

        return None

    async def download_and_process_image(self, image_url: str) -> str:
        """下载并处理图片"""
        logger.info(f"下载图片: {image_url[:100]}...")

        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    raise Exception(f"图片下载失败: HTTP {response.status}")
                img_data = await response.read()

        # 处理图片
        img = PILImage.open(BytesIO(img_data))

        # 调整大小（最大1024px）
        if max(img.size) > 1024:
            ratio = 1024 / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, PILImage.LANCZOS)

        # 转换为JPEG并编码为base64
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info(f"图片处理完成，大小: {len(base64_data)} 字符")
        return base64_data

    async def call_animetrace_api(self, img_base64: str, model: str) -> dict:
        """使用base64调用AnimeTrace API"""
        payload = {"base64": img_base64, "is_multi": 1, "model": model, "ai_detect": 0}

        model_name_map = {
            "pre_stable": "动漫识别模型",
            "full_game_model_kira": "GalGame识别模型", 
            "animetrace_high_beta": "通用识别模型"
        }
        logger.info(f"调用API - 模型: {model_name_map.get(model, model)} (base64方式)")

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, data=payload, timeout=30) as response:
                if response.status != 200:
                    await response.text()
                    raise Exception(f"API错误: HTTP {response.status}")

                result = await response.json()
                logger.info(f"API返回: {len(result.get('data', []))} 个结果")
                return result

    async def call_animetrace_api_with_url(self, image_url: str, model: str) -> dict:
        """使用URL直接调用AnimeTrace API"""
        payload = {"url": image_url, "is_multi": 1, "model": model, "ai_detect": 0}

        model_name_map = {
            "pre_stable": "动漫识别模型",
            "full_game_model_kira": "GalGame识别模型", 
            "animetrace_high_beta": "通用识别模型"
        }
        logger.info(f"调用API - 模型: {model_name_map.get(model, model)} (URL方式)")

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, data=payload, timeout=30) as response:
                if response.status != 200:
                    error_text = await response.text()
                    # 如果URL方式失败，返回空结果让上层逻辑回退到base64方式
                    if response.status == 422:
                        logger.info("URL识别失败，准备回退到base64方式")
                        return {"data": []}
                    raise Exception(f"API错误: HTTP {response.status}")

                result = await response.json()
                logger.info(f"API返回: {len(result.get('data', []))} 个结果")
                return result

    def format_results(self, data: dict, model: str) -> str:
        """格式化识别结果"""
        if not data.get("data") or not data["data"]:
            return "🔍 未找到匹配的信息"

        first_result = data["data"][0]
        characters = first_result.get("character", [])

        if not characters:
            return "🔍 未识别到具体角色信息"

        model_name_map = {
            "pre_stable": "动漫识别",
            "full_game_model_kira": "GalGame识别", 
            "animetrace_high_beta": "通用识别"
        }
        emoji_map = {
            "pre_stable": "🎌",
            "full_game_model_kira": "🎮", 
            "animetrace_high_beta": "🔍"
        }
        model_name = model_name_map.get(model, "图片识别")
        emoji = emoji_map.get(model, "🔍")

        lines = [f"**{emoji} {model_name}结果**", "=" * 20]

        # 显示前5个结果
        for i, char in enumerate(characters[:5]):
            name = char.get("character", "未知角色")
            work = char.get("work", "未知作品")
            lines.append(f"{i + 1}. **{name}** - 《{work}》")

        if len(characters) > 5:
            lines.append(f"\n> 共 {len(characters)} 个结果，显示前5项")

        lines.append("\n💡 数据来源: AnimeTrace，仅供参考")

        return "\n".join(lines)

    async def timeout_check(self, user_id: str):
        """30秒超时检查"""
        try:
            await asyncio.sleep(30)  # 等待30秒
            if user_id in self.waiting_sessions:
                # 30秒后仍然在等待，发送超时消息
                session = self.waiting_sessions[user_id]
                event = session["event"]
                del self.waiting_sessions[user_id]
                del self.timeout_tasks[user_id]
                try:
                    await event.send(event.plain_result("⏰ 识别请求已超时，请重新发送命令"))
                    logger.info(f"用户 {user_id} 的图片识别请求已超时")
                except Exception as send_error:
                    logger.warning(f"发送超时消息失败: {send_error}")
                    # 如果发送超时消息失败，记录日志但不影响清理操作
        except asyncio.CancelledError:
            # 任务被取消，说明用户已经发送了图片
            pass
        except Exception as e:
            logger.error(f"超时检查任务异常: {str(e)}")

    async def terminate(self):
        logger.info("动漫/Gal/二游识别插件已卸载")
        # 取消所有超时任务
        for task in self.timeout_tasks.values():
            task.cancel()
        self.timeout_tasks.clear()
