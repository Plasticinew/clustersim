from bisect import bisect_left, insort_left


class SortedDict:
    """Minimal drop-in replacement for the subset used by this repo."""

    def __init__(self):
        self._data = {}
        self._keys = []

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._keys)

    def __setitem__(self, key, value):
        if key not in self._data:
            insort_left(self._keys, key)
        self._data[key] = value

    def __delitem__(self, key):
        if key not in self._data:
            raise KeyError(key)
        del self._data[key]
        index = bisect_left(self._keys, key)
        if index >= len(self._keys) or self._keys[index] != key:
            raise KeyError(key)
        self._keys.pop(index)

    def popitem(self, index=-1):
        if not self._keys:
            raise KeyError('popitem(): dictionary is empty')
        if index < 0:
            index += len(self._keys)
        key = self._keys.pop(index)
        value = self._data.pop(key)
        return key, value

    def peekitem(self, index=-1):
        if not self._keys:
            raise KeyError('peekitem(): dictionary is empty')
        if index < 0:
            index += len(self._keys)
        key = self._keys[index]
        return key, self._data[key]
