import gzip
from typing import TypeVar, Union

from fastapi import Request, Response
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def gzip_response_if_requested(
    request: Request,
    response: T,
) -> Union[Response, T]:
    accept_encoding = request.headers.get("Accept-Encoding")
    if not accept_encoding or "gzip" not in accept_encoding:
        return response

    json_bytes = response.json().encode("utf-8")
    compressed = gzip.compress(json_bytes)

    response = Response(
        content=compressed,
        headers={
            "Content-Encoding": "gzip",
            "Content-Length": str(len(compressed)),
        },
    )
    return response
