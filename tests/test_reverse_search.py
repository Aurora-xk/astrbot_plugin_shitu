"""Offline checks for real uploaded-image search and text-first results."""

from support import load_plugin_package

load_plugin_package()


import asyncio
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image as PILImage
from astrbot.api.message_components import Image, Reply
from astrbot_plugin_shitu import reverse_search as backend
from astrbot_plugin_shitu.main import AnimeTracePlugin
from test_image_tool import Event

YANDEX_HTML = b'''<html><a href="https://irrelevant.example/">generic navigation</a>
<li class="CbirSites-Item"><div class="CbirSites-ItemThumb"><a href="https://images.example/portrait.png">1200x1440</a></div>
<div><a href="https://example.org/characters/hero?utm_source=test">Example Hero</a>
<p>Keyboardist and vocalist for Example Band.</p></div></li></html>'''

SAUCE_METADATA = b'''<div class="result"><div class="resultsimilarityinfo">89.20%</div>
<div class="resultmiscinfo"><a href="https://danbooru.donmai.us/posts/123">index</a></div>
<div class="resultcontent"><div class="resulttitle"><strong>Creator: </strong>artist-name<br/></div>
<div class="resultcontentcolumn"><strong>Source: </strong><a href="https://example.com/original">original</a><br/>
<strong>Material: </strong>example series<br/>example sequel<br/></div>
<div class="resultcontentcolumn"><strong>Characters: </strong><br/>example hero<br/></div></div></div>'''


class ParserTests(unittest.TestCase):
    def test_uploaded_search_sites_yields_text_evidence_and_source(self):
        result = backend.parse_yandex(YANDEX_HTML)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Example Hero")
        self.assertIn("Example Band", result[0]["snippet"])
        self.assertEqual(result[0]["url"], "https://example.org/characters/hero")
        self.assertEqual(result[0]["confidence"], "not_scored")

    def test_generic_keyword_page_never_counts_as_reverse_search(self):
        self.assertEqual(backend.parse_yandex(b'<a href="https://example.com/">a text-search result</a>'), [])

    def test_weak_similarity_is_discarded_and_candidates_are_marked(self):
        html = b'''<div class="result"><div class="resultsimilarityinfo">54.2%</div><div class="resultcontent"><a href="https://example.com/wrong">wrong</a></div></div>
        <div class="result"><div class="resultsimilarityinfo">92.1%</div><div class="resulttitle">Title</div><div class="resultcontent"><a href="https://example.com/art">art</a></div></div>'''
        rows = backend.parse_saucenao(html)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["similarity"], 92.1)
        self.assertEqual(rows[0]["confidence"], "candidate")

    def test_urls_do_not_echo_credentials_or_tracking(self):
        self.assertEqual(backend.clean_url('https://user:password@example.com/'), '')
        self.assertEqual(backend.clean_url('file:///secret'), '')
        self.assertEqual(backend.clean_url('https://example.com/a?id=3&token=FAKESECRET&utm_source=x'), 'https://example.com/a?id=3')

    def test_character_work_creator_and_index_are_not_lost(self):
        row = backend.parse_saucenao(SAUCE_METADATA)[0]
        self.assertEqual(row['characters'], 'example hero')
        self.assertEqual(row['material'], 'example series; example sequel')
        self.assertEqual(row['creator'], 'artist-name')
        self.assertIn('Characters: example hero', row['snippet'])
        self.assertEqual(row['url'], 'https://example.com/original')
        self.assertEqual(row['additional_urls'], ['https://danbooru.donmai.us/posts/123'])


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_limited_parser_runs_real_subprocess(self):
        result = await backend.parse_limited(YANDEX_HTML, "parse_yandex", 5)
        self.assertEqual(result[0]["title"], "Example Hero")

    async def test_strong_sauce_metadata_avoids_yandex_upload(self):
        result = {"status": "ok", "engine": "SauceNAO", "matches": backend.parse_saucenao(SAUCE_METADATA), "upload_succeeded": True}
        with patch.object(backend, "saucenao_search", AsyncMock(return_value=result)), \
             patch.object(backend, "yandex_search", AsyncMock()) as fallback:
            got = await backend.reverse_search(b"jpeg", 5)
        self.assertEqual(got['recognition_status'], 'metadata_candidate')
        self.assertEqual(got['matches'][0]['characters'], 'example hero')
        self.assertEqual(got['engine'], 'SauceNAO')
        fallback.assert_not_awaited()

    async def test_unavailable_engine_uses_one_free_fallback_without_error_leak(self):
        with patch.object(backend, "saucenao_search", AsyncMock(side_effect=RuntimeError("FAKESECRET"))), \
             patch.object(backend, "yandex_search", AsyncMock(return_value={"status": "partial", "engine": "Yandex Images", "matches": backend.parse_yandex(YANDEX_HTML)})) as fallback:
            result = await backend.reverse_search(b"jpeg")
        self.assertEqual(fallback.await_count, 1)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['recognition_status'], 'unconfirmed')
        self.assertNotIn("FAKESECRET", json.dumps(result))

    async def test_weak_sauce_match_continues_and_generic_pages_are_unconfirmed(self):
        weak = {"title": "weak candidate", "similarity": 70, "url": "https://example.com/weak"}
        with patch.object(backend, "saucenao_search", AsyncMock(return_value={"status": "ok", "engine": "SauceNAO", "matches": [weak]})), \
             patch.object(backend, "yandex_search", AsyncMock(return_value={"status": "partial", "engine": "Yandex Images", "matches": backend.parse_yandex(YANDEX_HTML)})) as fallback:
            result = await backend.reverse_search(b"jpeg")
        fallback.assert_awaited_once()
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['recognition_status'], 'unconfirmed')
        self.assertEqual(len(result['matches']), 2)

    async def test_no_match_is_not_replaced_by_guessed_name(self):
        with patch.object(backend, "yandex_search", AsyncMock(return_value={"status": "no_match", "engine": "Yandex Images", "matches": [], "upload_succeeded": True})), \
             patch.object(backend, "saucenao_search", AsyncMock(return_value={"status": "no_match", "engine": "SauceNAO", "matches": []})):
            result = await backend.reverse_search(b"jpeg")
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["matches"], [])
        self.assertIn("文字结果", result["guidance"])

    async def test_invalid_input_does_not_upload_and_errors_are_fixed(self):
        with patch.object(backend, "yandex_search", AsyncMock()) as upload:
            result = await backend.reverse_search(b"", True)
        self.assertEqual(result["error_code"], "invalid_reverse_image_input")
        upload.assert_not_awaited()
        with patch.object(backend, "yandex_search", AsyncMock(side_effect=RuntimeError("FAKEKEY"))), \
             patch.object(backend, "saucenao_search", AsyncMock(side_effect=RuntimeError("FAKEKEY"))):
            result = await backend.reverse_search(b"jpeg")
        self.assertEqual(result["error_code"], "reverse_search_unavailable")
        self.assertNotIn("FAKEKEY", json.dumps(result))


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = AnimeTracePlugin(SimpleNamespace(), {})
        self.plugin._session = object()

    async def test_tool_only_accepts_message_image_index_not_path_or_url(self):
        self.assertEqual(list(inspect.signature(self.plugin.reverse_image_search).parameters), ["event", "image_index", "max_results", "reference_message_id"])
        self.assertEqual(json.loads(await self.plugin.reverse_image_search(Event()))["status"], "no_image")

    async def test_real_component_uploaded_once_and_cache_avoids_repeated_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            PILImage.new("RGB", (32, 32), "white").save(path)
            self.plugin._image_bridge.local_roots = (Path(directory).resolve(),)
            event = Event([Image.fromFileSystem(str(path))])
            with patch("astrbot_plugin_shitu.main.upload_reverse_search", AsyncMock(return_value={"status": "ok", "matches": [{"title": "text result", "url": "https://example.com/"}]})) as upload:
                first = json.loads(await self.plugin.reverse_image_search(event))
                second = json.loads(await self.plugin.reverse_image_search(event))
        self.assertEqual(upload.await_count, 1)
        self.assertTrue(upload.call_args.args[0].startswith(b'\xff\xd8'))
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])

    async def test_cached_result_keeps_current_quote_selection_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.png"
            PILImage.new("RGB", (32, 32), "white").save(path)
            self.plugin._image_bridge.local_roots = (Path(directory).resolve(),)
            image = Image.fromFileSystem(str(path))
            result = {"status": "ok", "matches": [{"url": "https://example.org/art"}]}
            with patch("astrbot_plugin_shitu.main.upload_reverse_search", AsyncMock(return_value=result)) as upload:
                first = json.loads(await self.plugin.reverse_image_search(Event([image])))
                quoted = Event([Reply(id="10", chain=[image])])
                second = json.loads(await self.plugin.reverse_image_search(quoted))
        self.assertEqual(upload.await_count, 1)
        self.assertEqual(first["image_source"], "current_message")
        self.assertEqual(second["image_source"], "explicit_reply")
        self.assertTrue(second["cached"])

    async def test_prompt_prioritizes_text_and_actual_reverse_search(self):
        req = SimpleNamespace(system_prompt="persona")
        await self.plugin.image_identity_guidance(Event([Image.fromURL("https://example.com/image.png")]), req)
        self.assertIn("reverse_image_search", req.system_prompt)
        self.assertIn("characters/material/creator", req.system_prompt)
        self.assertIn("继续用 reverse_image_search", req.system_prompt)

    async def test_command_includes_textual_identity_instead_of_just_links(self):
        event = Event()
        event.plain_result = lambda text: text
        event.send = AsyncMock()
        result = {"status": "ok", "engine": "SauceNAO", "matches": backend.parse_saucenao(SAUCE_METADATA)}
        with patch.object(self.plugin, 'reverse_image_search', AsyncMock(return_value=json.dumps(result))):
            await self.plugin.reverse_image_command(event)
        answer = event.send.call_args.args[0]
        self.assertIn('角色标注：example hero', answer)
        self.assertIn('作品标注：example series', answer)
        self.assertIn('作者标注：artist-name', answer)


class UpstreamPresentationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = AnimeTracePlugin(SimpleNamespace(), {})
        self.event = Event()
        self.event.plain_result = lambda text: text
        self.event.chain_result = lambda chain: chain
        self.event.send = AsyncMock()

    async def test_v41_card_path_remains_the_default(self):
        results = {"data": [{"not_confident": False, "character": [{"character": "Synthetic Hero"}]}]}
        crops = [{"index": 1, "characters": results["data"][0]["character"],
                  "crop_path": "/unused-synthetic-crop.png"}]
        with patch("astrbot_plugin_shitu.main.collect_crops", AsyncMock(return_value=crops)), \
             patch.object(self.plugin, "_render_result_card", AsyncMock(return_value="/unused-synthetic-card.png")) as render:
            await self.plugin.send_combined_result(self.event, "synthetic.png", results, "synthetic-model")
        render.assert_awaited_once_with("synthetic-model", crops)
        sent = self.event.send.call_args.args[0]
        self.assertEqual(len(sent), 1)
        self.assertIsInstance(sent[0], Image)

    async def test_v41_plain_result_omits_unknown_work_when_cropping_fails(self):
        results = {"data": [{"character": [{"character": "Synthetic Hero", "work": "未知作品"}]}]}
        with patch("astrbot_plugin_shitu.main.collect_crops", AsyncMock(return_value=[])):
            await self.plugin.send_combined_result(self.event, "synthetic.png", results, "synthetic-model")
        sent = self.event.send.call_args.args[0]
        self.assertIn("Synthetic Hero", sent)
        self.assertNotIn("未知作品", sent)


if __name__ == "__main__": unittest.main()
