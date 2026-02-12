import urllib

from inference.core.env import LICENSE_SERVER
from urllib.parse import quote


def wrap_url(url: str) -> str:
    if not LICENSE_SERVER:
        return url
    return f"http://{LICENSE_SERVER}/proxy?url=" + quote(url, safe="~()*!'")
