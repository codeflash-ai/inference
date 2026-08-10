class WorkflowMemoryCache:
    cache = {}

    @classmethod
    def get_dict(cls, namespace):
        try:
            # Fast path: direct read (dict __getitem__ faster than not-in check)
            return cls.cache[namespace]
        except KeyError:
            # Only create if truly missing (avoid double lookup)
            value = {}
            cls.cache[namespace] = value
            return value

    @classmethod
    def clear_namespace(cls, namespace):
        if namespace in WorkflowMemoryCache.cache:
            del WorkflowMemoryCache.cache[namespace]
