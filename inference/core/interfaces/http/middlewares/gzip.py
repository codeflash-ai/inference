import gzip
from typing import TypeVar, Union

from fastapi import Request, Response
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def gzip_response_if_requested(
    request: Request,
    response: T,
) -> Union[Response, T]:
    if "gzip" not in request.headers.get("Accept-Encoding", ""):
        return response
    json_content = response.json().encode("utf-8")
    compressed_body = gzip.compress(json_content)
    response = Response(
        content=compressed_body,
        headers={
            "Content-Encoding": "gzip",
            "Content-Length": str(len(compressed_body)),
        },
    )
    return response
