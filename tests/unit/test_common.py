from nomad_tools.common import cached_property


class A:
    def __init__(self):
        self.i = 100

    @cached_property
    def get(self) -> int:
        self.i += 100
        return self.i


def test_common_cached_property():
    a = A()
    assert a.i == 100
    assert a.get == 200
    assert a.get == 200
    assert a.get == 200
    assert a.get == 200
