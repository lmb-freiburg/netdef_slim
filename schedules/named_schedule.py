class NamedSchedule():
    def __init__(self, name, max_iter):
        self._name = name
        self._max_iter = max_iter

    def name(self):
        return self._name

    def max_iter(self):
        return self._max_iter

