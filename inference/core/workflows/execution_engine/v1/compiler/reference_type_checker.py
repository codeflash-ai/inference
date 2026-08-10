from typing import List, Union

from inference.core.workflows.errors import ReferenceTypeError
from inference.core.workflows.execution_engine.entities.types import Kind


def validate_reference_kinds(
    expected: List[Union[Kind, str]],
    actual: List[Union[Kind, str]],
    error_message: str,
) -> None:
    # Use generator expressions with set constructor for less overhead
    expected_kind_names = set(_get_kind_name(e) for e in expected)
    actual_kind_names = set(_get_kind_name(a) for a in actual)
    # Optimize the "*" check by checking both sets simultaneously before intersection computation
    if "*" in expected_kind_names or "*" in actual_kind_names:
        return None
    # Use direct boolean check for intersection (faster than computing length of set)
    if not expected_kind_names & actual_kind_names:
        raise ReferenceTypeError(
            public_message=error_message,
            context="workflow_compilation | execution_graph_construction",
        )


def _get_kind_name(kind: Union[Kind, str]) -> str:
    # Use direct attribute access without intermediate isinstance lookup table
    if type(kind) is Kind:
        return kind.name
    return kind
