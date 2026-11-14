from support import load_plugin_package
load_plugin_package()

"""Verify upstream card output locally, including preserved confidence warnings."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image as PILImage
from astrbot.api.message_components import Image, Plain
from astrbot_plugin_shitu.main import AnimeTracePlugin
from astrbot_plugin_shitu.crops import make_crops
from astrbot_plugin_shitu.card import render_card


class UpstreamCardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        original=tempfile.mkdtemp
        self.patch=patch('astrbot_plugin_shitu.main.tempfile.mkdtemp',
                         side_effect=lambda **kwargs:original(dir=self.tmp.name,**kwargs))
        self.patch.start();self.addCleanup(self.patch.stop)
        self.path=Path(self.tmp.name)/'source.png'
        PILImage.new('RGB',(80,100),'blue').save(self.path)
        self.item=dict(box=[0,0,1,1],not_confident=True,
                       character=[{'character':'测试角色','work':'测试作品'}])
        self.plugin=object.__new__(AnimeTracePlugin)
        self.plugin._models=[];self.plugin._session=None
        self.plugin.max_characters_per_role=5;self.plugin.result_as_image=True
        self.plugin.no_result_reply='no result'
        self.event=SimpleNamespace(send=AsyncMock(),chain_result=lambda chain:chain,plain_result=lambda text:text)

    def test_local_crop_and_chinese_card_render(self):
        items=make_crops(PILImage.open(self.path),[self.item])
        self.assertEqual(len(items),1)
        image=render_card('test model',items)
        self.assertGreater(image.width,100)
        output=Path(self.tmp.name)/'card.png';image.save(output)
        self.assertEqual(PILImage.open(output).format,'PNG')

    async def test_card_reply_keeps_low_confidence_text(self):
        await self.plugin.send_combined_result(self.event,str(self.path),{'data':[self.item]},'test')
        chain=self.event.send.call_args.args[0]
        self.assertTrue(any(isinstance(c,Image) for c in chain))
        self.assertTrue(any(isinstance(c,Plain) and '低置信' in c.text for c in chain))

    async def test_crop_text_fallback_keeps_low_confidence_text(self):
        self.plugin.result_as_image=False
        await self.plugin.send_combined_result(self.event,str(self.path),{'data':[self.item]},'test')
        chain=self.event.send.call_args.args[0]
        self.assertTrue(any(isinstance(c,Image) for c in chain))
        self.assertTrue(any(isinstance(c,Plain) and '低置信' in c.text for c in chain))

    async def test_missing_confidence_stays_uncertain_in_text_output(self):
        text = self.plugin.format_results({"data": [{"character": [
            {"character": "Synthetic", "work": "Test"}]}]}, "synthetic")
        self.assertIn("未提供置信标记", text)
        self.assertIn("仅为候选", text)
