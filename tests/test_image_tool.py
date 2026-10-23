"""Offline contract tests against AstrBot's installed Image/Reply components."""

from support import load_plugin_package

load_plugin_package()


import asyncio
import base64
import inspect
import json
import sys
import tempfile
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image as PILImage
from astrbot.api.message_components import Image, Plain, Reply

from astrbot_plugin_shitu.image_tool import ImageRecognitionBridge, MAX_BYTES, bounded_results


class Event:
    def __init__(self, messages=(), sender="member-a", origin="qq:group:g1", group="g1", platform="bot-a", bot=None):
        self.messages = list(messages)
        self.sender = sender
        self.unified_msg_origin = origin
        self.group = group
        self.platform = platform
        self.bot = bot
        self.message_obj = SimpleNamespace(self_id="self")

    def get_messages(self): return self.messages
    def get_sender_id(self): return self.sender
    def get_platform_id(self): return self.platform
    def get_platform_name(self): return "aiocqhttp"
    def get_group_id(self): return self.group


class Bot:
    def __init__(self, data):
        self.data = data
        self.calls = []

    async def call_action(self, **kwargs):
        self.calls.append(kwargs)
        return self.data


class Content:
    def __init__(self, chunks): self.chunks = chunks
    async def iter_chunked(self, n):
        for chunk in self.chunks:
            yield chunk


class Response:
    status = 200
    def __init__(self, chunks, length=None):
        self.content = Content(chunks)
        self.content_length = length
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.png"
        PILImage.new("RGB", (32, 32), "purple").save(self.path)
        self.image = Image.fromFileSystem(str(self.path))
        self.now = 1000
        self.bridge = ImageRecognitionBridge(clock=lambda: self.now,
                                             local_roots=[self.directory.name])
        self.requests = []

    async def service(self, jpeg, model):
        self.requests.append((jpeg, model))
        await asyncio.sleep(0)
        return {"data": [{"box": [0, 0, 1, 1], "not_confident": True,
                          "character": [{"character": "Synthetic candidate", "work": "Test work"}]}]}

    async def call(self, event=None, **kwargs):
        return await self.bridge.recognize(event or Event([self.image]), "model-a", None,
                                            kwargs.pop("caller", self.service), **kwargs)

    async def test_recent_isolated_by_sender_session_and_platform(self):
        self.bridge.remember(Event([self.image]))
        for event in [Event(sender="member-b"), Event(origin="qq:group:g2", group="g2"),
                      Event(platform="bot-b"), Event(origin="qq:private:member-a", group="")]:
            images, origin = await self.bridge.select(event)
            self.assertEqual(images, [])
        images, origin = await self.bridge.select(Event())
        self.assertEqual(len(images), 1)
        self.assertEqual(origin, "same_sender_recent_5m")

    async def test_recent_expires_without_refresh_on_text(self):
        self.bridge.remember(Event([self.image]))
        self.now += 299
        self.assertEqual(len((await self.bridge.select(Event([Plain("who?")])))[0]), 1)
        self.now += 1
        self.assertEqual((await self.bridge.select(Event()))[0], [])

    async def test_explicit_reply_precedes_recent_and_never_guesses_fallback(self):
        self.bridge.remember(Event([self.image]))
        quoted = Image.fromURL("https://example.com/quoted.png")
        event = Event([Reply(id="5", chain=[quoted], sender_id="member-b")])
        images, origin = await self.bridge.select(event)
        self.assertEqual(images[0].file, quoted.file)
        self.assertEqual(origin, "explicit_reply")
        self.assertEqual((await self.bridge.select(Event([Reply(id="6", chain=[])])))[0], [])

    async def test_empty_quote_uses_exact_native_id_in_same_group(self):
        bot = Bot({"message_id": 5, "group_id": "g1", "message": [
            {"type": "image", "data": {"file": "image.png", "url": "https://example.com/p.png"}}]})
        images, origin = await self.bridge.select(Event([Reply(id="5", chain=[])], bot=bot))
        self.assertEqual(len(images), 1)
        self.assertEqual(bot.calls, [{"action": "get_msg", "message_id": 5}])
        bot.data["group_id"] = "g2"
        self.assertEqual((await self.bridge.select(Event([Reply(id="5", chain=[])], bot=bot)))[0], [])
        bot.data["group_id"] = "g1"
        bot.data["message_id"] = 6
        self.assertEqual((await self.bridge.select(Event([Reply(id="5", chain=[])], bot=bot)))[0], [])

    async def test_private_native_quote_rejects_other_peer(self):
        bot = Bot({"message_id": 5, "message_type": "private", "sender": {"user_id": "member-b"},
                   "message": [{"type": "image", "data": {"url": "https://example.com/p.png"}}]})
        event = Event([Reply(id="5", chain=[])], bot=bot, group="", origin="qq:private:member-a")
        self.assertEqual((await self.bridge.select(event))[0], [])
        bot.data["sender"]["user_id"] = "member-a"
        self.assertEqual(len((await self.bridge.select(event))[0]), 1)

    async def test_result_preserves_uncertainty_box_and_no_confirmation(self):
        result = await self.call()
        self.assertEqual(result["status"], "candidates")
        self.assertTrue(result["roles"][0]["not_confident"])
        self.assertEqual(result["roles"][0]["box"], [0, 0, 1, 1])
        self.assertIn("不要", result["guidance"])
        jpeg, model = self.requests[0]
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))

    async def test_hash_cache_coalesces_concurrent_requests_and_expires(self):
        first, second = await asyncio.gather(self.call(), self.call())
        self.assertEqual(len(self.requests), 1)
        self.assertEqual({first["cached"], second["cached"]}, {True, False})
        self.now += 901
        await self.call()
        self.assertEqual(len(self.requests), 2)

    async def test_preprocessing_is_serial_across_different_images(self):
        original = self.bridge._prepare
        counts = {"active": 0, "peak": 0}
        guard = threading.Lock()
        def prepare(raw):
            with guard:
                counts["active"] += 1
                counts["peak"] = max(counts["peak"], counts["active"])
            try:
                time.sleep(0.02)
                return original(raw)
            finally:
                with guard: counts["active"] -= 1
        self.bridge._prepare = prepare
        events = []
        for index, color in enumerate(["green", "purple", "blue"]):
            path = Path(self.directory.name) / (str(index) + ".png")
            PILImage.new("RGB", (32, 32), color).save(path)
            events.append(Event([Image.fromFileSystem(str(path))], sender=str(index)))
        await asyncio.gather(*(self.call(event) for event in events))
        self.assertEqual(counts["peak"], 1)

    async def test_cancel_waits_for_thread_before_releasing_worker(self):
        entered = asyncio.Event()
        release = threading.Event()
        calls = []
        original = self.bridge._prepare
        loop = asyncio.get_running_loop()
        def prepare(raw):
            calls.append(True)
            if len(calls) == 1:
                loop.call_soon_threadsafe(entered.set)
                release.wait(3)
            return original(raw)
        self.bridge._prepare = prepare
        first = asyncio.create_task(self.call())
        await asyncio.wait_for(entered.wait(), 2)
        first.cancel()
        second = asyncio.create_task(self.call())
        try:
            await asyncio.sleep(0.02)
            self.assertFalse(first.done())
            self.assertTrue(self.bridge.lock.locked())
            self.assertEqual(len(calls), 1)
            first.cancel()  # A second cancellation must not abandon the thread either.
            await asyncio.sleep(0.02)
            self.assertFalse(first.done())
            self.assertEqual(len(calls), 1)
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual((await asyncio.wait_for(second, 2))["status"], "candidates")
        self.assertEqual(len(calls), 2)

    async def test_download_failure_cached_without_repeating_remote_request(self):
        class FailingSession:
            def __init__(self): self.calls = 0
            def get(self, *args, **kwargs):
                self.calls += 1
                raise RuntimeError("transport secret")
        session = FailingSession()
        event = Event([Image.fromURL("https://example.com/failure.png")])
        first = await self.bridge.recognize(event, "test", session, self.service)
        second = await self.bridge.recognize(event, "test", session, self.service)
        self.assertEqual(first["error_code"], "image_unavailable")
        self.assertTrue(second["cached"])
        self.assertEqual(session.calls, 1)
        self.assertNotIn("secret", json.dumps(first))

    async def test_cached_service_failure_redacted_and_not_retried(self):
        calls = []
        async def failed(jpeg, model):
            calls.append(model)
            raise RuntimeError("secret token=should-never-appear")
        first = await self.call(caller=failed)
        second = await self.call(caller=failed)
        self.assertEqual(first["error_code"], "recognition_service_unavailable")
        self.assertTrue(second["cached"])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("secret", json.dumps(first))
        self.now += 91
        await self.call(caller=failed)
        self.assertEqual(len(calls), 2)

    async def test_different_image_cooldown_no_second_upload(self):
        await self.call()
        other = Path(self.directory.name) / "second.png"
        PILImage.new("RGB", (32, 32), "green").save(other)
        result = await self.call(Event([Image.fromFileSystem(str(other))]))
        self.assertEqual(result["status"], "cooldown")
        self.assertEqual(len(self.requests), 1)

    async def test_invalid_local_image_and_outside_root_never_upload(self):
        text = Path(self.directory.name) / "secret.txt"
        text.write_text("not an image")
        result = await self.call(Event([Image.fromFileSystem(str(text))]))
        self.assertEqual(result["error_code"], "invalid_image")
        result = await self.call(Event([Image(file="file:///etc/hosts")]))
        self.assertEqual(result["error_code"], "image_source_not_allowed")
        self.assertEqual(self.requests, [])

    async def test_default_webchat_attachment_root_accepts_images_rejects_config_and_escape(self):
        from unittest.mock import patch
        data = Path(self.directory.name) / "data"
        attachments = data / "attachments"
        attachments.mkdir(parents=True)
        image = attachments / "uploaded.png"
        PILImage.new("RGB", (32, 32), "purple").save(image)
        config = data / "cmd_config.json"
        config.write_text('{"synthetic": "config must never be read"}')
        outside_image = data / "private.png"
        PILImage.new("RGB", (32, 32), "green").save(outside_image)
        escaped = attachments / "escape.png"
        escaped.symlink_to(outside_image)
        with patch("astrbot_plugin_shitu.image_tool.get_astrbot_data_path", return_value=str(data)):
            bridge = ImageRecognitionBridge(clock=lambda: self.now)
        result = await bridge.recognize(Event([Image.fromFileSystem(str(image))]),
                                        "test", None, self.service)
        self.assertEqual(result["status"], "candidates")
        for path in (config, outside_image, escaped):
            result = await bridge.recognize(Event([Image.fromFileSystem(str(path))]),
                                            "test", None, self.service)
            self.assertEqual(result["error_code"], "image_source_not_allowed")
        self.assertEqual(len(self.requests), 1)

    async def test_image_index_input_and_no_image_do_not_call_service(self):
        for value in [0, 5, True, "1", "https://example.com"]:
            self.assertEqual((await self.call(image_index=value))["error_code"], "invalid_image_index")
        self.assertEqual((await self.call(Event()))["status"], "no_image")
        self.assertEqual(self.requests, [])

    async def test_remote_chunks_read_until_eof_and_not_forwarded_as_url(self):
        raw = self.path.read_bytes()
        session = Session(Response([raw[:16], raw[16:31], raw[31:]]))
        component = Image.fromURL("https://example.com/image.png")
        received = await self.bridge._image_bytes(component, session)
        self.assertEqual(received, raw)
        self.assertFalse(session.calls[0][1]["allow_redirects"])
        self.assertEqual(self.bridge._prepare(received)[:2], b"\xff\xd8")

    async def test_remote_size_and_private_literal_rejected(self):
        response = Response([b"x" * 10], MAX_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "image_too_large"):
            await self.bridge._image_bytes(Image.fromURL("https://example.com/x.png"), Session(response))
        for url in ["http://127.0.0.1/x.png", "http://[::1]/x.png", "http://localhost/x.png"]:
            with self.assertRaisesRegex(ValueError, "image_source_not_allowed"):
                await self.bridge._image_bytes(Image.fromURL(url), Session(response))

    async def test_base64_validates_and_service_works(self):
        image = Image.fromBase64(base64.b64encode(self.path.read_bytes()).decode())
        result = await self.call(Event([image]))
        self.assertEqual(result["status"], "candidates")

    async def test_memory_caches_bounded(self):
        for i in range(150):
            self.bridge.remember(Event([self.image], sender=str(i)))
        self.assertEqual(len(self.bridge.recent), self.bridge.capacity)

    async def test_recent_byte_budget_and_oversized_inline_do_not_retain_stale_image(self):
        self.bridge.remember(Event([self.image]))
        huge = Image.fromBase64("a" * (self.bridge.recent_entry_budget + 1))
        self.bridge.remember(Event([huge]))
        self.assertNotIn(self.bridge._key(Event()), self.bridge.recent)
        long_image = Image.fromURL("https://example.com/" + "a" * 7000)
        for i in range(150):
            self.bridge.remember(Event([long_image], sender=str(i)))
        self.assertLess(sum(item[2] for item in self.bridge.recent.values()),
                        self.bridge.recent_byte_budget + 1)
        self.assertLess(len(self.bridge.recent), self.bridge.capacity)

    async def test_transparent_image_composited_over_white(self):
        output = BytesIO()
        PILImage.new("RGBA", (32, 32), (0, 0, 0, 0)).save(output, format="PNG")
        converted = self.bridge._prepare(output.getvalue())
        with PILImage.open(BytesIO(converted)) as image:
            self.assertTrue(all(channel > 250 for channel in image.getpixel((0, 0))))


class ResultTests(unittest.TestCase):

    def test_missing_confidence_and_empty_results_stay_uncertain(self):
        result = bounded_results({"data": [{"character": [{"character": "candidate"}]}]})
        self.assertIsNone(result["roles"][0]["not_confident"])
        self.assertEqual(bounded_results({"data": []})["status"], "no_match")
        with self.assertRaises(ValueError): bounded_results({"message": "unexpected"})

    def test_truncates_text_ignores_malformed_boxes(self):
        result = bounded_results({"data": [{"box": [0, float("nan"), 1, 1], "not_confident": "false",
                                           "character": [{"character": "x" * 500, "work": "w" * 500}]}]})
        role = result["roles"][0]
        self.assertIsNone(role["box"])
        self.assertIsNone(role["not_confident"])
        self.assertEqual(len(role["candidates"][0]["character"]), 160)


class PluginContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from astrbot_plugin_shitu.main import AnimeTracePlugin
        self.plugin = AnimeTracePlugin(SimpleNamespace(), {})

    async def test_tool_only_exposes_index_no_url_or_path(self):
        signature = inspect.signature(self.plugin.identify_anime_character)
        self.assertEqual(list(signature.parameters), ["event", "image_index"])
        self.assertEqual(signature.parameters["image_index"].default, 1)

    async def test_llm_hook_idempotent_and_no_network(self):
        req = SimpleNamespace(system_prompt="original persona")
        event = Event([Image.fromURL("https://example.com/synthetic.png")])
        await self.plugin.image_identity_guidance(event, req)
        first = req.system_prompt
        await self.plugin.image_identity_guidance(event, req)
        self.assertEqual(req.system_prompt, first)
        self.assertTrue(first.startswith("original persona"))
        self.assertIn("identify_anime_character", first)
        plain_req = SimpleNamespace(system_prompt="different session")
        await self.plugin.image_identity_guidance(Event(sender="member-b"), plain_req)
        self.assertEqual(plain_req.system_prompt, "different session")


    async def test_bounded_api_parses_fragmented_response_and_errors_are_fixed(self):
        class PostSession:
            def __init__(self, response): self.response = response
            def post(self, *args, **kwargs): return self.response
        raw = json.dumps({"code": 0, "data": []}).encode()
        self.plugin._session = PostSession(Response([raw[:4], raw[4:15], raw[15:]]))
        self.assertEqual(await self.plugin._call_bounded_image_api(b"jpeg", "test"), {"code": 0, "data": []})
        response = Response([b"password=never-read"])
        response.status = 401
        self.plugin._session = PostSession(response)
        with self.assertRaisesRegex(ValueError, "^recognition_service_unavailable$"):
            await self.plugin._call_bounded_image_api(b"jpeg", "test")


    async def test_empty_models_do_not_index_error_or_retry_immediately(self):
        self.plugin._model_cache_time = asyncio.get_running_loop().time()
        self.assertIsNone(await self.plugin._get_default_model())


if __name__ == "__main__":
    unittest.main()
