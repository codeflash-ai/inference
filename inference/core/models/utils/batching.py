import itertools
from typing import Generator, Iterable, List, TypeVar

B = TypeVar("B")


def create_batches(
    sequence: Iterable[B], batch_size: int
) -> Generator[List[B], None, None]:
    batch_size = max(batch_size, 1)
    iterator = iter(sequence)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch
