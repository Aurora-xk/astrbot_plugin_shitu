from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Image as MsgImage, Reply
import astrbot.api.message_components as Comp
import aiohttp
import asyncio
import base64
import re
from io import BytesIO
from PIL import Image as PILImage
import os
import tempfile


@register("astrbot_plugin_shitu", "aurora", "动漫/Gal/二游图片识别插件", "3.5", "https://github.com/Aurora-xk/astrbot_plugin_shitu")
class AnimeTracePlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.api_url = "https://api.animetrace.com/v1/search"
        self.waiting_sessions = {}  # 简单的会话管理
        self.timeout_tasks = {}  # 存储超时任务
        
        # 加载配置
        if config:
            shitu_config = config.get("shitu_settings", {})
        else:
            shitu_config = getattr(self.context, '_config', {}).get("shitu_settings", {})
        
        self.timeout_seconds = shitu_config.get("timeout_seconds", 30)
        self.prompt_send_image = shitu_config.get("prompt_send_image", "📷 请发送要识别的图片（30秒内有效）")
        self.prompt_timeout = shitu_config.get("prompt_timeout", "⏰ 识别请求已超时，请重新发送命令")
        self.return_crops = shitu_config.get("return_crops", True)
        self.max_crops = shitu_config.get("max_crops", 5)

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

    @filter.command("头像动漫识别")
    async def avatar_anime_search(self, event: AstrMessageEvent, args=None):
        """识别QQ用户头像（动漫模型）"""
        return await self.handle_avatar_recognition(event, "pre_stable")

    @filter.command("头像gal识别")
    async def avatar_gal_search(self, event: AstrMessageEvent, args=None):
        """识别QQ用户头像（GalGame模型）"""
        return await self.handle_avatar_recognition(event, "full_game_model_kira")

    @filter.command("头像识别")
    async def avatar_trace_search(self, event: AstrMessageEvent, args=None):
        """识别QQ用户头像（通用模型）"""
        return await self.handle_avatar_recognition(event, "animetrace_high_beta")

    async def handle_image_recognition(self, event: AstrMessageEvent, model: str):
        """简化的图片识别处理"""
        user_id = event.get_sender_id()

        # 检查当前消息是否包含图片（包括引用消息中的图片）
        image_url = await self.extract_image_from_event(event)
        if image_url:
            # 如果找到图片，直接进行识别
            await self.process_image_recognition(event, image_url, model)
            return

        # 检查是否是引用消息但没有图片的情况
        try:
            raw_event = event._event if hasattr(event, "_event") else event
            if hasattr(raw_event, "reply_to_message") and raw_event.reply_to_message:
                logger.debug("检测到引用消息，但引用消息中没有找到图片")
                await event.send(event.plain_result("❌ 引用消息中没有找到图片，请确保引用的消息包含图片"))
                return
        except Exception as e:
            logger.warning(f"检查引用消息状态时出错: {str(e)}")

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

        await event.send(event.plain_result(self.prompt_send_image))
        logger.debug(f"用户 {user_id} 进入等待图片状态，等待{self.timeout_seconds}秒")

    async def handle_avatar_recognition(self, event: AstrMessageEvent, model: str):
        """处理QQ头像识别"""
        try:
            # 提取被@的用户或手动输入的QQ号
            mentioned_user_id = await self.extract_mentioned_user(event)

            if not mentioned_user_id:
                # 如果没有@任何人，默认使用发送者自己的头像
                mentioned_user_id = event.get_sender_id()
                await event.send(event.plain_result("📸 识别您自己的头像..."))
            else:
                # 检查是否是手动输入的QQ号（通过正则匹配确认）
                messages = event.get_messages()
                full_text = ""
                for msg in messages:
                    if hasattr(msg, "text"):
                        full_text += str(msg.text)
                    elif hasattr(msg, "type") and msg.type == "Plain":
                        full_text += str(msg)

                import re
                qq_match = re.search(r"头像(?:动漫|gal)?识别\s*(\d{5,12})", full_text)
                if qq_match and qq_match.group(1) == mentioned_user_id:
                    await event.send(event.plain_result(f"📸 识别QQ号 {mentioned_user_id} 的头像..."))

            # 获取头像URL
            avatar_url = f"https://q.qlogo.cn/headimg_dl?dst_uin={mentioned_user_id}&spec=640"
            # 标记此事件已被处理，避免消息监听器重复处理
            event._avatar_command_processed = True

            # 识别头像
            await self.process_image_recognition(event, avatar_url, model)

        except Exception as e:
            logger.error(f"头像识别失败: {str(e)}")
            await event.send(event.plain_result(f"❌ 头像识别失败: {str(e)}"))

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听所有消息，处理等待中的图片识别请求和特殊格式的头像识别命令"""
        user_id = event.get_sender_id()

        # 检查特殊格式的头像识别命令（消息中包含@但命令可能被遗漏的情况）
        messages = event.get_messages()
        full_text = ""
        for msg in messages:
            if hasattr(msg, "text"):
                full_text += str(msg.text)
            elif hasattr(msg, "type") and msg.type == "Plain":
                full_text += str(msg)

        # 只有当标准命令处理器未处理时才检查
        if not hasattr(event, "_avatar_command_processed"):
            avatar_patterns = [
                (r"头像动漫识别", "pre_stable"),
                (r"头像gal识别", "full_game_model_kira"),
                (r"头像识别", "animetrace_high_beta"),
            ]

            for pattern, model in avatar_patterns:
                if re.search(pattern, full_text):
                    # 标记为已处理，避免重复
                    event._avatar_command_processed = True
                    await self.handle_avatar_recognition(event, model)
                    return  # 处理完后直接返回，避免重复处理

        # 检查用户是否在等待图片识别
        if user_id not in self.waiting_sessions:
            return

        session = self.waiting_sessions[user_id]

        # 检查是否超时
        current_time = asyncio.get_event_loop().time()
        if current_time - session["timestamp"] > self.timeout_seconds:
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
        """处理图片识别，并在识别后返回裁剪图片"""
        try:
            # 首先尝试直接使用URL调用API（更高效）
            results = await self.call_animetrace_api_with_url(image_url, model)

            # 如果URL方式失败，再回退到下载图片方式
            if not results or not results.get("data"):
                logger.debug("URL识别方式未返回结果，尝试下载图片识别...")
                img_data = await self.download_and_process_image(image_url)
                results = await self.call_animetrace_api(img_data, model)

            # 将所有内容（图片+文字）合并到一条消息中发送
            await self.send_combined_result(event, image_url, results, model)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"识别失败: {error_msg}")

            # 更友好的错误提示
            if "HTTP 500" in error_msg:
                user_msg = "❌ 识别服务暂时不可用，请稍后重试"
            elif "HTTP 422" in error_msg:
                user_msg = "❌ 图片格式不支持，请尝试其他图片"
            elif "timeout" in error_msg.lower():
                user_msg = "❌ 识别超时，请稍后重试"
            else:
                user_msg = f"❌ 识别失败: {error_msg}"

            try:
                await event.send(event.plain_result(user_msg))
            except Exception as send_error:
                logger.warning(f"发送错误消息失败: {send_error}")
                # 如果错误消息也发送失败，记录日志但不抛出异常

    async def extract_mentioned_user(self, event: AstrMessageEvent) -> str:
        """从事件中提取被@的用户QQ号或手动输入的QQ号"""
        messages = event.get_messages()

        # 首先检查是否有手动输入的QQ号
        full_text = ""
        for msg in messages:
            if hasattr(msg, "text"):
                full_text += str(msg.text)
            elif hasattr(msg, "type") and msg.type == "Plain":
                full_text += str(msg)

        # 匹配手动输入QQ号的格式：头像识别 12345678910 或 头像识别12345678910
        import re
        qq_match = re.search(r"头像(?:动漫|gal)?识别\s*(\d{5,12})", full_text)
        if qq_match:
            qq_number = qq_match.group(1)
            return qq_number

        for msg in messages:
            # 检查是否有@提及
            if hasattr(msg, "type") and msg.type == "At":
                # QQ平台的@消息
                if hasattr(msg, "qq"):
                    return str(msg.qq)
                if hasattr(msg, "user_id"):
                    return str(msg.user_id)

            # 检查文本中的@格式
            if hasattr(msg, "text"):
                text = str(msg.text)
                # 匹配 [CQ:at,qq=123456] 格式
                at_match = re.search(r"\[CQ:at,qq=(\d+)\]", text)
                if at_match:
                    return at_match.group(1)

                # 匹配 @用户名 格式（需要平台支持）
                # 有些平台会直接解析为At组件，这里作为备选

        return None

    async def extract_image_from_event(self, event: AstrMessageEvent) -> str:
        """从事件中提取图片URL"""
        messages = event.get_messages()

        # 首先检查当前消息中的图片
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

        # 检查引用消息中的图片（Telegram等平台）
        try:
            # 查找Reply组件
            for msg in messages:
                if isinstance(msg, Reply):
                    # Reply组件包含原始消息的信息
                    if hasattr(msg, "chain") and msg.chain:
                        # 在引用消息的chain中查找图片
                        for reply_msg in msg.chain:
                            if isinstance(reply_msg, MsgImage):
                                if hasattr(reply_msg, "url") and reply_msg.url:
                                    return reply_msg.url.strip()
                                if hasattr(reply_msg, "file") and reply_msg.file:
                                    file_content = str(reply_msg.file)
                                    if "http" in file_content:
                                        import re
                                        urls = re.findall(r"https?://[^\s\`\']+", file_content)
                                        if urls:
                                            return urls[0].strip("`'")

        except Exception as e:
            logger.warning(f"检查引用消息图片时出错: {str(e)}")

        return None

    async def download_and_process_image(self, image_url: str) -> str:
        """下载并处理图片"""
        logger.debug(f"下载图片: {image_url[:100]}...")

        try:
            # 处理Telegram的特殊URL格式
            if image_url.startswith("telegram://"):
                file_id = image_url.replace("telegram://", "")
                logger.debug(f"检测到Telegram文件，file_id: {file_id}")
                # 对于Telegram文件，我们需要通过file_id获取实际的文件URL
                # 这里简化处理，直接返回一个标识，让上层逻辑处理
                # 在实际环境中，需要调用Telegram Bot API获取文件路径
                # Telegram文件现在支持识别，继续正常处理流程

            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=30) as response:
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

            logger.debug(f"图片处理完成，大小: {len(base64_data)} 字符")
            return base64_data
        except asyncio.TimeoutError:
            raise Exception("图片下载超时，请稍后重试")
        except Exception as e:
            logger.error(f"图片处理失败: {str(e)}")
            raise Exception(f"图片处理失败: {str(e)}")

    async def call_animetrace_api(self, img_base64: str, model: str) -> dict:
        """使用base64调用AnimeTrace API"""
        payload = {"base64": img_base64, "is_multi": 1, "model": model, "ai_detect": 0}

        model_name_map = {
            "pre_stable": "动漫识别模型",
            "full_game_model_kira": "GalGame识别模型",
            "animetrace_high_beta": "通用识别模型"
        }
        logger.debug(f"调用API - 模型: {model_name_map.get(model, model)} (base64方式)")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, data=payload, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.warning(f"API返回错误状态: HTTP {response.status}, 响应: {error_text[:200]}")
                        raise Exception(f"API错误: HTTP {response.status}")

                    result = await response.json()
                    logger.debug(f"API返回: {len(result.get('data', []))} 个结果")
                    return result
        except asyncio.TimeoutError:
            logger.error("API调用超时")
            raise Exception("识别服务响应超时，请稍后重试")
        except Exception as e:
            logger.error(f"base64 API调用失败: {str(e)}")
            raise

    async def call_animetrace_api_with_url(self, image_url: str, model: str) -> dict:
        """使用URL直接调用AnimeTrace API"""
        payload = {"url": image_url, "is_multi": 1, "model": model, "ai_detect": 0}

        model_name_map = {
            "pre_stable": "动漫识别模型",
            "full_game_model_kira": "GalGame识别模型",
            "animetrace_high_beta": "通用识别模型"
        }
        logger.debug(f"调用API - 模型: {model_name_map.get(model, model)} (URL方式)")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, data=payload, timeout=30) as response:
                    if response.status != 200:
                        # 如果URL方式失败，返回空结果让上层逻辑回退到base64方式
                        if response.status in [422, 500, 502, 503, 504]:
                            logger.debug(f"URL识别失败 (HTTP {response.status})，准备回退到base64方式")
                            return {"data": []}
                        raise Exception(f"API错误: HTTP {response.status}")

                    result = await response.json()
                    logger.debug(f"API返回: {len(result.get('data', []))} 个结果")
                    return result
        except Exception as e:
            logger.warning(f"URL方式调用失败: {str(e)}，准备回退到base64方式")
            return {"data": []}

    def format_results(self, data: dict, model: str) -> str:
        """格式化识别结果"""
        if not data.get("data") or not data["data"]:
            return "🔍 未找到匹配的信息"

        results = data["data"]
        # 过滤掉没有角色信息的框
        results = [item for item in results if item.get("character")]
        if not results:
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

        # 纯文本格式输出
        lines = [f"{emoji} {model_name}结果"]

        for idx, item in enumerate(results, start=1):
            characters = item.get("character", [])
            if not characters:
                continue

            if len(results) > 1:
                lines.append(f"\n第 {idx} 个角色：")

            for i, char in enumerate(characters[:5]):
                name = char.get("character", "未知角色")
                work = char.get("work", "未知作品")
                lines.append(f"{i + 1}. {name} - 《{work}》")

            if len(characters) > 5:
                lines.append(f"共 {len(characters)} 个结果，显示前5项")

        lines.append("数据来源: AnimeTrace，仅供参考")

        return "\n".join(lines)

    async def send_combined_result(self, event: AstrMessageEvent, image_url: str, results: dict, model: str):
        """将所有图片和文字合并到一条消息中发送"""
        try:
            data_list = results.get("data") or []
            if not data_list:
                # 没有识别结果，只发送文字
                response = self.format_results(results, model)
                await event.send(event.plain_result(response))
                return

            # 构建消息链
            chain = []
            
            # 如果需要发送裁剪图片，先处理图片
            if self.return_crops:
                # 下载原图
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url, timeout=30) as response:
                        if response.status != 200:
                            logger.debug(f"裁剪图片下载失败: HTTP {response.status}")
                            # 下载失败，只发送文字结果
                            response_text = self.format_results(results, model)
                            await event.send(event.plain_result(response_text))
                            return
                        img_data = await response.read()

                img = PILImage.open(BytesIO(img_data)).convert("RGB")
                w, h = img.size

                # 临时目录，存放裁剪后的图片
                tmp_dir = tempfile.mkdtemp(prefix="astrbot_shitu_crops_")
                crop_paths = []

                # 裁剪所有角色的图片
                for idx, item in enumerate(data_list, start=1):
                    if len(crop_paths) >= self.max_crops:
                        break

                    box = item.get("box")
                    if not box or len(box) != 4:
                        continue

                    # 参考 anime/animetrace_search.py 的裁剪逻辑
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

                # 将图片和对应的文字添加到消息链中
                for idx, out_path, item in crop_paths:
                    # 添加图片
                    chain.append(Comp.Image.fromFileSystem(out_path))
                    
                    # 添加该角色的文字信息
                    characters = item.get("character") or []
                    if characters:
                        text_lines = []
                        if len(crop_paths) > 1:
                            text_lines.append(f"第 {idx} 个角色：")
                        
                        for i, char in enumerate(characters[:5]):
                            name = char.get("character", "未知角色")
                            work = char.get("work", "未知作品")
                            text_lines.append(f"{i + 1}. {name} - 《{work}》")
                        
                        if len(characters) > 5:
                            text_lines.append(f"共 {len(characters)} 个结果，显示前5项")
                        
                        if text_lines:
                            chain.append(Comp.Plain("\n".join(text_lines)))
                            chain.append(Comp.Plain(""))  # 添加空行分隔

            # 添加总览文字（如果有多于一个角色，或者没有发送裁剪图片）
            if not self.return_crops or len(crop_paths) < len(data_list):
                response_text = self.format_results(results, model)
                chain.append(Comp.Plain(response_text))
            else:
                # 只添加数据来源说明
                chain.append(Comp.Plain("💡 数据来源: AnimeTrace，仅供参考"))

            # 发送合并后的消息
            if chain:
                await event.send(event.chain_result(chain))
            else:
                # 如果消息链为空，发送文字结果
                response_text = self.format_results(results, model)
                await event.send(event.plain_result(response_text))

        except Exception as e:
            logger.warning(f"发送合并结果失败: {e}")
            # 失败时回退到只发送文字结果
            try:
                response_text = self.format_results(results, model)
                await event.send(event.plain_result(response_text))
            except Exception as send_error:
                logger.warning(f"发送文字结果也失败: {send_error}")

    async def timeout_check(self, user_id: str):
        """超时检查"""
        try:
            await asyncio.sleep(self.timeout_seconds)  # 等待配置的超时时间
            if user_id in self.waiting_sessions:
                # 超时后仍然在等待，发送超时消息
                session = self.waiting_sessions[user_id]
                event = session["event"]
                del self.waiting_sessions[user_id]
                del self.timeout_tasks[user_id]
                try:
                    await event.send(event.plain_result(self.prompt_timeout))
                    logger.debug(f"用户 {user_id} 的图片识别请求已超时")
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
