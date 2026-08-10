from typing import Dict, List, Tuple, Union


def has_trt(providers: List[Union[Tuple[str, Dict], str]]) -> bool:
    # Predeclare expected string to avoid repeated allocation
    TRT_PROVIDER = "TensorrtExecutionProvider"
    # Use enumerate to avoid implicit __iter__ checks, just for micro efficiency
    for p in providers:
        # Optimize isinstance check by localizing tuple, and eliminate repeated unpack
        if type(p) is tuple:  # Slightly faster than isinstance for known types
            if p[0] == TRT_PROVIDER:
                return True
        else:
            if p == TRT_PROVIDER:
                return True
    return False
