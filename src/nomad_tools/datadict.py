import copy
from typing import Any, ChainMap, Union


def all_annotations(cls) -> ChainMap:
    """Returns a dictionary-like ChainMap that includes annotations for all
    attributes defined in cls or inherited from superclasses."""
    return ChainMap(
        *(c.__annotations__ for c in cls.__mro__ if "__annotations__" in c.__dict__)
    )


def _init_value(name: str, aname: str, atype: Any, val: Any):
    # print(f"Constructing {aname} with type {atype} from {val}")
    msg = f"When constructing class {name} expected type {atype} for variable {name}, but received {type(val)} with value: {val}"
    origin = getattr(atype, "__origin__", None)
    if origin == list:
        assert type(val) == origin, msg
        return [atype.__args__[0](x) for x in val]
    elif origin == dict:
        assert type(val) == origin, msg
        return {atype.__args__[0](k): atype.__args__[1](v) for k, v in val.items()}
    elif origin == Union:
        if type(val) in atype.__args__:
            return val
        assert len(atype.__args__) == 2 and atype.__args__[1] == type(
            None
        ), f"Only Optional handled"
        return _init_value(name, aname, atype.__args__[0], val)
    elif issubclass(atype, DataDict):
        return atype(val)
    assert type(val) == atype, msg
    return val


def _asdict_value(fname: str, val: Any):
    if isinstance(val, list):
        return [_asdict_value(fname, x) for x in val]
    elif isinstance(val, dict):
        return {
            _asdict_value(fname, k): _asdict_value(fname, v) for k, v in val.items()
        }
    elif hasattr(val, fname):
        return getattr(val, fname)()
    return val


class DataDict:
    """
    Data dictionary - a mix between dataclass and AttrDict from stackoverflow.
    Basically a dictionary with standard dictionary accesseses proxied to self.__dict__.
    Then additional __init__ function that allows constructing from a dictionary and will
    also construct any nested AttrDict objects from type hints.
    """

    def __init__(self, *args, **kwargs):
        data = dict(*args, **kwargs)
        annotations = all_annotations(self.__class__)
        # Copy default values from class.
        for akey, atype in annotations.items():
            if akey in self.__class__.__dict__:
                self.__dict__[akey] = copy.deepcopy(self.__class__.__dict__[akey])
        # Intialize values from dictionary.
        for key, val in data.items():
            if key in annotations:
                self.__dict__[key] = _init_value(
                    self.__class__.__name__, key, annotations[key], val
                )
            else:
                self.__dict__[key] = val
        self.__post_init__()

    def __post_init__(self):
        pass

    def __setitem__(self, k, v):
        self.__dict__[k] = v

    def __getitem__(self, k):
        return self.__dict__[k]

    def get(self, k, v=None):
        return self.__dict__.get(k, v)

    def setdefault(self, k, v):
        return self.__dict__.setdefault(k, v)

    def __contains__(self, k):
        return k in self.__dict__

    def __delitem__(self, k):
        del self.__dict__[k]

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def asdict(self):
        fname = self.asdict.__name__
        return {k: _asdict_value(fname, v) for k, v in self.__dict__.items()}

    def __repr__(self):
        data = " ".join(f"{k}={self.__dict__[k]!r}" for k in sorted(self.__dict__))
        return f"{self.__class__.__name__}({data})"

    def __eq__(self, o):
        if self.__class__ == o.__class__:
            return self.__dict__ == o.__dict__
        else:
            return False
