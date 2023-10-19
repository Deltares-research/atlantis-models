from enum import Enum


class Foo(Enum):
    bar = 1

    def __getitem__(self, item):
        return self[item].value
