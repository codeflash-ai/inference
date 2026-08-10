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
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = value
