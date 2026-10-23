"""Load this checkout as a package, without changing AstrBot or its components."""

from pathlib import Path
import sys
import types

import aiohttp


async def _network_disabled(*args, **kwargs):
    raise AssertionError("Offline tests must not make external HTTP requests")


def load_plugin_package():
    aiohttp.ClientSession._request = _network_disabled
    name = "astrbot_plugin_shitu"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        sys.modules[name] = package
