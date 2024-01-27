from typing import Any, Dict, List, Optional

from nomad_tools.nomadlib.datadict import DataDict


class Simple(DataDict):
    var: int


def test_datadict_simple():
    a = Simple({"var": 123})
    assert "alist" not in a
    assert a.var == 123
    assert a.asdict() == {"var": 123}
    a.var = 234
    assert a.asdict() == {"var": 234}
    assert a.var == 234
    assert str(a) == "Simple(var=234)"
    c = [Simple(), Simple()]
    assert str(c) == "[Simple(), Simple()]"


class Collect(DataDict):
    alist: List[Simple]
    adict: Dict[int, Simple]
    aopt: Optional[Simple]
    aopt2: Optional[Simple] = None
    var2: int = -1


def test_datadict_collect():
    init = {
        "alist": [
            {"var": 1},
            {"var": 2},
        ],
        "adict": {
            1: {"var": 3},
            2: {"var": 4},
        },
        "aopt": {"var": 5},
    }
    b = Collect(init)
    assert b.alist == [Simple(var=1), Simple(var=2)]
    assert b["alist"] == b.alist
    assert b.adict == {1: Simple(var=3), 2: Simple(var=4)}
    assert b["adict"] == b.adict
    assert b.aopt == Simple(var=5)
    assert b.aopt2 is None
    assert b.var2 == -1
    assert b.asdict() == {**init, "aopt2": None, "var2": -1}


class Inherit(Simple):
    var2: int = 1


def test_datadict_inherit():
    c = Inherit({"var": 123})
    assert c.var == 123
    assert c["var"] == 123
    assert c.var2 == 1
    assert c.asdict() == {"var": 123, "var2": 1}
    c.var2 = 234
    assert c.asdict() == {"var": 123, "var2": 234}


class Nested(DataDict):
    var: Optional[Dict[str, Simple]]


def test_datadict_nested():
    d = Nested()
    assert "var" not in d
    d = Nested({"var": None})
    assert "var" in d
    assert d.var is None
    d = Nested({"var": {"key": {"var": 1}}})
    assert d.var
    assert d.var["key"] == Simple(var=1)


class ToList(DataDict):
    alist: List[Simple]


def test_datadict_tolist():
    init = {
        "alist": [
            {"var": 1},
        ]
    }
    e = ToList(init)
    assert e.asdict() == init


class ToDict(DataDict):
    alist: Dict[int, Simple]


def test_datadict_todict():
    init = {
        1: {"var": 1},
    }
    f = ToDict(init)
    assert f.asdict() == init


def test_datadict_any():
    class A(DataDict):
        a: List[Any]
        b: Optional[List[List[Any]]]

    init = {
        "a": ["a", 1, None],
        "b": None,
    }
    assert A(init).asdict() == init
    init2 = {
        "a": ["a", 1, None],
        "b": [["b", 2], ["c", 3]],
    }
    assert A(init2).asdict() == init2


class StrMember(DataDict):
    ID: str

    def str(self):
        return f"{self.ID}"


def test_datadict_strmember():
    v = StrMember(dict(ID=1))
    assert v.str() == f"{v.ID}"
