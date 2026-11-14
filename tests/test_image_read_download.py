"""Offline image-read regressions using real components and bounded fake HTTP."""

import asyncio
import json
import tempfile
import unittest
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
from PIL import Image as PILImage
from astrbot.api.message_components import Image

from astrbot_plugin_shitu import main as plugin_module
from astrbot_plugin_shitu.image_tool import MAX_BYTES, ImageReadError, ImageRecognitionBridge
from astrbot_plugin_shitu.main import AnimeTracePlugin


SIGNED_URL = "https://cdn.example.test/image.png?signature=FAKE_SECRET_TOKEN&key=PRIVATE_QUERY"


def png_bytes(color="white"):
    output = BytesIO()
    PILImage.new("RGB", (32, 24), color).save(output, format="PNG")
    return output.getvalue()


class Stream:
    def __init__(self, parts):
        self.parts = parts
        self.reads = 0

    async def iter_chunked(self, size):
        self.reads += 1
        for part in self.parts:
            await asyncio.sleep(0)
            if isinstance(part, BaseException):
                raise part
            yield part


class Response:
    def __init__(self, parts=(), status=200, content_length=None):
        self.status = status
        self.content_length = content_length
        self.content = Stream(parts)


class Request:
    def __init__(self, response):
        self.response = response
        self.exited = False

    async def __aenter__(self):
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    async def __aexit__(self, *args):
        self.exited = True
        return False


class Session:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.requests = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        request = Request(self.responses.pop(0))
        self.requests.append(request)
        return request


class Event:
    def __init__(self, component):
        self.components = [component]
        self.extra = {}
        self.message_obj = SimpleNamespace(message_id="synthetic-image-message")
        self.unified_msg_origin = "qq:group:synthetic-group"

    def get_messages(self):
        return self.components

    def get_platform_id(self):
        return "synthetic-bot"

    def get_sender_id(self):
        return "synthetic-member"

    def set_extra(self, key, value):
        self.extra[key] = value

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)


class ImageReadDownloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "allowed"
        self.root.mkdir()
        self.now = 1000.0
        self.bridge = ImageRecognitionBridge(clock=lambda: self.now, local_roots=[self.root])
        self.raw = png_bytes()
        self.path = self.root / "original.png"
        self.path.write_bytes(self.raw)

    async def read(self, session, url=SIGNED_URL):
        return await self.bridge._image_bytes(Image.fromURL(url), session)

    def assert_private(self, value):
        text = str(value)
        for secret in (SIGNED_URL, "FAKE_SECRET_TOKEN", "PRIVATE_QUERY", "secret-body", "proxy-password"):
            self.assertNotIn(secret, text)

    async def test_same_component_local_path_precedes_remote_and_unbounded_converter(self):
        component = Image(file=SIGNED_URL, url=SIGNED_URL, path=str(self.path))
        session = Session()
        with patch.object(Image, "convert_to_file_path", AsyncMock(side_effect=AssertionError("unbounded converter"))) as convert:
            self.assertEqual(await self.bridge._image_bytes(component, session), self.raw)
        self.assertEqual(session.calls, [])
        convert.assert_not_awaited()

    async def test_same_component_file_uri_precedes_url(self):
        component = Image(file=self.path.as_uri(), url=SIGNED_URL)
        session = Session()
        with patch.object(Image, "convert_to_file_path", AsyncMock(side_effect=AssertionError("unbounded converter"))) as convert:
            self.assertEqual(await self.bridge._image_bytes(component, session), self.raw)
        self.assertEqual(session.calls, [])
        convert.assert_not_awaited()

    async def test_missing_optional_local_path_falls_back_to_bounded_remote(self):
        component = Image(file=SIGNED_URL, url=SIGNED_URL, path=str(self.root / "missing.png"))
        session = Session(Response([self.raw]))
        self.assertEqual(await self.bridge._image_bytes(component, session), self.raw)
        self.assertEqual(len(session.calls), 1)

    async def test_outside_optional_path_does_not_leak_local_contents(self):
        outside = Path(self.directory.name) / "private.png"
        outside.write_bytes(b"secret-body")
        component = Image(file=SIGNED_URL, url=SIGNED_URL, path=str(outside))
        session = Session(Response([self.raw]))
        self.assertEqual(await self.bridge._image_bytes(component, session), self.raw)
        self.assertEqual(len(session.calls), 1)

    async def test_symlink_to_outside_optional_file_falls_back_remote(self):
        outside = Path(self.directory.name) / "private.png"
        outside.write_bytes(b"secret-body")
        link = self.root / "link.png"
        link.symlink_to(outside)
        component = Image(file=link.as_uri(), url=SIGNED_URL)
        session = Session(Response([self.raw]))
        self.assertEqual(await self.bridge._image_bytes(component, session), self.raw)
        self.assertEqual(len(session.calls), 1)

    async def test_oversized_optional_local_file_falls_back_remote(self):
        oversized = self.root / "large.png"
        with oversized.open("wb") as stream:
            stream.truncate(MAX_BYTES + 1)
        component = Image(file=SIGNED_URL, url=SIGNED_URL, path=str(oversized))
        session = Session(Response([self.raw]))
        self.assertEqual(await self.bridge._image_bytes(component, session), self.raw)
        self.assertEqual(len(session.calls), 1)

    async def test_local_only_component_reads_without_network_or_converter(self):
        session = Session()
        with patch.object(Image, "convert_to_file_path", AsyncMock(side_effect=AssertionError("unbounded converter"))) as convert:
            self.assertEqual(await self.bridge._image_bytes(Image.fromFileSystem(str(self.path)), session), self.raw)
        self.assertEqual(session.calls, [])
        convert.assert_not_awaited()

    async def test_outside_local_only_source_is_rejected_safely(self):
        outside = Path(self.directory.name) / "private.png"
        outside.write_bytes(b"secret-body")
        with self.assertRaises((ImageReadError, ValueError)) as caught:
            await self.bridge._image_bytes(Image.fromFileSystem(str(outside)), Session())
        self.assert_private(caught.exception)

    async def test_exact_signed_url_reuses_complete_bytes(self):
        session = Session(Response([self.raw[:20], self.raw[20:]]))
        first = await self.read(session)
        second = await self.read(session)
        self.assertEqual(first, self.raw)
        self.assertEqual(second, self.raw)
        self.assertEqual(len(session.calls), 1)

    async def test_distinct_signed_queries_never_share_bytes(self):
        other = png_bytes("red")
        session = Session(Response([self.raw]), Response([other]))
        self.assertEqual(await self.read(session), self.raw)
        self.assertEqual(await self.read(session, SIGNED_URL.replace("FAKE_SECRET_TOKEN", "OTHER_TOKEN")), other)
        self.assertEqual(len(session.calls), 2)

    async def test_byte_cache_expires_after_five_minutes(self):
        other = png_bytes("red")
        session = Session(Response([self.raw]), Response([other]))
        self.assertEqual(await self.read(session), self.raw)
        self.now += 301
        self.assertEqual(await self.read(session), other)
        self.assertEqual(len(session.calls), 2)

    async def test_clear_discards_download_bytes(self):
        session = Session(Response([self.raw]), Response([self.raw]))
        await self.read(session)
        self.bridge.clear()
        await self.read(session)
        self.assertEqual(len(session.calls), 2)

    async def test_byte_cache_entry_capacity_evicts_oldest(self):
        session = Session(*(Response([self.raw]) for _ in range(18)))
        for index in range(17):
            await self.read(session, f"https://cdn.example.test/{index}.png?token=synthetic")
        await self.read(session, "https://cdn.example.test/0.png?token=synthetic")
        self.assertEqual(len(session.calls), 18)

    async def test_byte_budget_evicts_old_download_even_below_entry_limit(self):
        large = b"x" * (6 * 1024 * 1024)
        session = Session(*(Response([large]) for _ in range(4)))
        for index in range(3):
            self.assertEqual(len(await self.read(session, f"https://cdn.example.test/{index}.png")), len(large))
        await self.read(session, "https://cdn.example.test/0.png")
        self.assertEqual(len(session.calls), 4)

    async def test_partial_stream_failure_is_not_cached_and_closes_response(self):
        session = Session(Response([self.raw[:20], aiohttp.ClientPayloadError("secret-body " + SIGNED_URL)]), Response([self.raw]))
        with self.assertRaises(ImageReadError) as caught:
            await self.read(session)
        self.assertEqual(caught.exception.code, "image_download_failed")
        self.assertEqual(caught.exception.stage, "download")
        self.assert_private(caught.exception)
        self.assertTrue(session.requests[0].exited)
        self.assertEqual(await self.read(session), self.raw)
        self.assertEqual(len(session.calls), 2)

    async def test_cancelled_download_propagates_and_is_not_cached(self):
        session = Session(Response([self.raw[:20], asyncio.CancelledError()]), Response([self.raw]))
        with self.assertRaises(asyncio.CancelledError):
            await self.read(session)
        self.assertTrue(session.requests[0].exited)
        self.assertEqual(await self.read(session), self.raw)
        self.assertEqual(len(session.calls), 2)

    async def test_content_length_limit_rejects_before_body_read(self):
        response = Response([self.raw], content_length=MAX_BYTES + 1)
        session = Session(response)
        with self.assertRaises(ImageReadError) as caught:
            await self.read(session)
        self.assertEqual(caught.exception.code, "image_too_large")
        self.assertEqual(response.content.reads, 0)

    async def test_stream_limit_enforces_bound_without_content_length(self):
        session = Session(Response([b"x" * MAX_BYTES, b"x"]))
        with self.assertRaises(ImageReadError) as caught:
            await self.read(session)
        self.assertEqual(caught.exception.code, "image_too_large")
        self.assertEqual(caught.exception.stage, "download")

    async def test_http_status_failure_keeps_safe_status_not_body(self):
        response = Response([b"secret-body"], status=403)
        with self.assertRaises(ImageReadError) as caught:
            await self.read(Session(response))
        self.assertEqual(caught.exception.code, "image_download_failed")
        self.assertEqual(caught.exception.stage, "download")
        self.assertEqual(caught.exception.http_status, 403)
        self.assertEqual(response.content.reads, 0)
        self.assert_private(caught.exception)

    async def test_http_200_body_timeout_preserves_status_and_retries_fresh(self):
        session = Session(Response([self.raw[:20], asyncio.TimeoutError(SIGNED_URL)]), Response([self.raw]))
        with self.assertRaises(ImageReadError) as caught:
            await self.read(session)
        self.assertEqual(caught.exception.code, "image_download_timeout")
        self.assertEqual(caught.exception.stage, "download")
        self.assertEqual(caught.exception.http_status, 200)
        self.assert_private(caught.exception)
        self.assertEqual(await self.read(session), self.raw)

    async def test_connect_timeout_has_no_fabricated_http_status(self):
        with self.assertRaises(ImageReadError) as caught:
            await self.read(Session(asyncio.TimeoutError("proxy-password " + SIGNED_URL)))
        self.assertEqual(caught.exception.code, "image_download_timeout")
        self.assertIsNone(caught.exception.http_status)
        self.assert_private(caught.exception)

    async def test_remote_timeout_policy_and_no_redirects_are_explicit(self):
        session = Session(Response([self.raw]))
        with patch.object(Image, "convert_to_file_path", AsyncMock(side_effect=AssertionError("unbounded converter"))) as convert:
            await self.read(session)
        options = session.calls[0][1]
        self.assertIs(options["allow_redirects"], False)
        timeout = options["timeout"]
        self.assertEqual((timeout.total, timeout.connect, timeout.sock_connect, timeout.sock_read), (45, 10, 10, 20))
        convert.assert_not_awaited()

    async def test_recognition_retains_typed_download_error_without_upload(self):
        caller = AsyncMock()
        session = Session(Response([asyncio.TimeoutError(SIGNED_URL)]))
        result = await self.bridge.recognize(Event(Image.fromURL(SIGNED_URL)), "synthetic-model", session, caller)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "image_download_timeout")
        self.assert_private(json.dumps(result))
        caller.assert_not_awaited()

    async def test_read_error_rejects_untrusted_metadata(self):
        error = ImageReadError(SIGNED_URL, "secret-body", True)
        self.assertEqual((error.code, error.stage, error.http_status), ("image_unavailable", "read", None))
        self.assert_private(error)


class ReverseSearchPhaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.raw = png_bytes()
        self.plugin = object.__new__(AnimeTracePlugin)
        self.plugin._image_bridge = ImageRecognitionBridge(local_roots=[self.directory.name])
        self.plugin._reverse_lock = asyncio.Lock()
        self.plugin._reverse_cache = OrderedDict()
        self.plugin._reverse_next_request = 0
        self.plugin._reverse_closed = False
        self.plugin._session = Session(Response([self.raw]))
        self.event = Event(Image.fromURL(SIGNED_URL))

    def assert_private(self, value):
        for secret in (SIGNED_URL, "FAKE_SECRET_TOKEN", "PRIVATE_QUERY", "secret-body", "proxy-password"):
            self.assertNotIn(secret, str(value))

    async def run_tool(self):
        return json.loads(await self.plugin.reverse_image_search(self.event))

    async def test_successfully_downloaded_image_is_real_jpeg_before_upload(self):
        with patch.object(plugin_module, "upload_reverse_search", AsyncMock(return_value={"status": "ok", "matches": []})) as upload:
            result = await self.run_tool()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["image_source"], "current_message")
        self.assertFalse(result["cached"])
        with PILImage.open(BytesIO(upload.call_args.args[0])) as decoded:
            self.assertEqual(decoded.format, "JPEG")
            self.assertEqual(decoded.size, (32, 24))

    async def test_read_timeout_returns_download_phase_and_does_not_upload(self):
        self.plugin._session = Session(Response([self.raw[:20], asyncio.TimeoutError(SIGNED_URL)]))
        with patch.object(plugin_module, "upload_reverse_search", AsyncMock()) as upload, patch.object(plugin_module.logger, "warning") as warning:
            result = await self.run_tool()
        self.assertEqual((result["error_code"], result["stage"], result["http_status"]), ("image_download_timeout", "download", 200))
        self.assertEqual(result["image_index"], 1)
        self.assertEqual(result["image_source"], "current_message")
        self.assertFalse(self.plugin._reverse_lock.locked())
        self.assertFalse(self.plugin._image_bridge.lock.locked())
        self.assertEqual(len(self.plugin._reverse_cache), 0)
        upload.assert_not_awaited()
        self.assert_private(json.dumps(result))
        self.assert_private(warning.call_args_list)
        self.assertIn("stage=download", str(warning.call_args_list))

    async def test_http_denial_returns_status_without_body_or_signed_url(self):
        self.plugin._session = Session(Response([b"secret-body"], status=410))
        with patch.object(plugin_module.logger, "warning") as warning:
            result = await self.run_tool()
        self.assertEqual((result["error_code"], result["stage"], result["http_status"]), ("image_download_failed", "download", 410))
        self.assert_private(json.dumps(result))
        self.assert_private(warning.call_args_list)

    async def test_invalid_image_is_prepare_failure_not_download_failure(self):
        self.plugin._session = Session(Response([b"not an image secret-body"]))
        with patch.object(plugin_module, "upload_reverse_search", AsyncMock()) as upload, patch.object(plugin_module.logger, "warning") as warning:
            result = await self.run_tool()
        self.assertEqual((result["error_code"], result["stage"]), ("invalid_image", "prepare"))
        upload.assert_not_awaited()
        self.assert_private(json.dumps(result))
        self.assert_private(warning.call_args_list)

    async def test_prepare_timeout_returns_accurate_stage(self):
        with patch.object(self.plugin._image_bridge, "_prepare_with_cancellation", AsyncMock(side_effect=asyncio.TimeoutError(SIGNED_URL))), patch.object(plugin_module, "upload_reverse_search", AsyncMock()) as upload, patch.object(plugin_module.logger, "warning") as warning:
            result = await self.run_tool()
        self.assertEqual((result["error_code"], result["stage"]), ("image_prepare_timeout", "prepare"))
        upload.assert_not_awaited()
        self.assert_private(json.dumps(result))
        self.assert_private(warning.call_args_list)

    async def test_upload_timeout_is_distinct_and_does_not_cache_failure(self):
        with patch.object(plugin_module, "upload_reverse_search", AsyncMock(side_effect=asyncio.TimeoutError("proxy-password " + SIGNED_URL))), patch.object(plugin_module.logger, "warning") as warning:
            result = await self.run_tool()
        self.assertEqual((result["error_code"], result["stage"]), ("reverse_search_timeout", "upload"))
        self.assertEqual(len(self.plugin._reverse_cache), 0)
        self.assert_private(json.dumps(result))
        self.assert_private(warning.call_args_list)

    async def test_upload_exception_does_not_leak_response_or_url(self):
        with patch.object(plugin_module, "upload_reverse_search", AsyncMock(side_effect=RuntimeError("secret-body " + SIGNED_URL))), patch.object(plugin_module.logger, "warning") as warning:
            result = await self.run_tool()
        self.assertEqual((result["error_code"], result["stage"]), ("reverse_image_search_failed", "upload"))
        self.assert_private(json.dumps(result))
        self.assert_private(warning.call_args_list)

    async def test_caller_cancellation_propagates_without_warning_and_unlocks(self):
        self.plugin._session = Session(Response([self.raw[:20], asyncio.CancelledError()]))
        with patch.object(plugin_module, "upload_reverse_search", AsyncMock()) as upload, patch.object(plugin_module.logger, "warning") as warning:
            with self.assertRaises(asyncio.CancelledError):
                await self.run_tool()
        self.assertFalse(self.plugin._reverse_lock.locked())
        self.assertFalse(self.plugin._image_bridge.lock.locked())
        self.assertEqual(len(self.plugin._reverse_cache), 0)
        upload.assert_not_awaited()
        warning.assert_not_called()

    async def _cancel_during(self, phase):
        entered = asyncio.Event()
        exited = asyncio.Event()

        async def waiting(*args):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                exited.set()

        target = self.plugin._image_bridge if phase == "prepare" else plugin_module
        name = "_prepare_with_cancellation" if phase == "prepare" else "upload_reverse_search"
        with patch.object(target, name, waiting), patch.object(plugin_module.logger, "warning") as warning:
            running = asyncio.create_task(self.run_tool())
            await asyncio.wait_for(entered.wait(), 2)
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(running, 2)
        self.assertTrue(exited.is_set())
        self.assertFalse(self.plugin._reverse_lock.locked())
        self.assertFalse(self.plugin._image_bridge.lock.locked())
        self.assertEqual(len(self.plugin._reverse_cache), 0)
        warning.assert_not_called()

    async def test_prepare_cancellation_propagates_and_unlocks(self):
        await self._cancel_during("prepare")

    async def test_upload_cancellation_stops_worker_and_unlocks(self):
        await self._cancel_during("upload")


if __name__ == "__main__":
    unittest.main()
