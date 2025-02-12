class NestedIterator:
    def __init__(self, nested_list):
        self.stack = list(reversed(nested_list))

    def __iter__(self):
        return self

    def __next__(self):
        while self.stack:
            current = self.stack.pop()
            if isinstance(current, list):
                self.stack.extend(reversed(current))
            else:
                return current
        raise StopIteration
