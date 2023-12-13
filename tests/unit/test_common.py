from nomad_tools.common import andjoin, cached_property


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

def test_andjoin():
    assert andjoin([]) == ""
    assert andjoin([1]) == "1"
    assert andjoin([1, 2]) == "1 and 2"
    assert andjoin([1, 2, 3]) == "1, 2 and 3"
    assert andjoin([1, 2, 3, 4]) == "1, 2, 3 and 4"
