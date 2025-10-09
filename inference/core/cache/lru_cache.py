import collections

from inference.core.logger import logger


class LRUCache:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def set_max_size(self, capacity):
        self.capacity = capacity
        self.enforce_size()

    def enforce_size(self):
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def get(self, key):
        cache = self.cache  # Local variable lookup is faster than attribute lookup
        # Fast path using move_to_end (added in Python 3.2) avoids removal and reinsertion
        try:
            value = cache[key]
            cache.move_to_end(key)
            return value
        except KeyError:
            return None

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            self.enforce_size()
        self.cache[key] = value
