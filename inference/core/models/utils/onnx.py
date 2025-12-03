from typing import Dict, List, Tuple, Union


def has_trt(providers: List[Union[Tuple[str, Dict], str]]) -> bool:
    for p in providers:
        if type(p) is tuple:
            if p[0] == "TensorrtExecutionProvider":
                return True
        elif p == "TensorrtExecutionProvider":
            return True
    return False
