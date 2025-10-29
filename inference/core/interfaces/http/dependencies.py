from typing import Optional, Union

from fastapi import Request
from starlette.datastructures import UploadFile

from inference.core.exceptions import InputImageLoadError


async def parse_body_content_for_legacy_request_handler(
    request: Request,
) -> Optional[Union[bytes, UploadFile]]:
    # Fetch Content-Type once
    headers = request.headers
    content_type = headers.get("Content-Type")
    if content_type is None:
        return None

    # Fast path: Only query for "image" parameter if not multipart
    if "multipart/form-data" in content_type:
        # Only call await request.form() when necessary
        form_data = await request.form()
        file_part = form_data.get("file")
        if file_part is None:
            raise InputImageLoadError(
                message="Expected image to be send in part named 'file' of multipart/form-data request",
                public_message="Expected image to be send in part named 'file' of multipart/form-data request",
            )
        return file_part

    # To avoid potentially expensive parsing of query params,
    # fetch only the 'image' parameter directly (fast path)
    # If possible, access the underlying dict for speed.
    query_params = request.query_params
    image_reference_in_query = None
    if hasattr(query_params, "_dict"):
        # Starlette's QueryParams often has a _dict attribute for O(1) lookup
        image_reference_in_query = query_params._dict.get("image")
    else:
        # fallback: normal mapping protocol
        image_reference_in_query = query_params.get("image")

    if content_type is None or image_reference_in_query:
        return None
    return await request.body()
