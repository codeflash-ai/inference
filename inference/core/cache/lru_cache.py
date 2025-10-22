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
        value = self.cache.get(key)
        if value is not None:
            # Move accessed key to end to maintain LRU order
            self.cache.move_to_end(key)
            return value
        return None

    def set(self, key, value):
        if self.cache.pop(key, None) is None:
            while len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value
