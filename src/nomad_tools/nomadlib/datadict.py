from __future__ import annotations

import copy
import enum
import logging
from typing import Any, ChainMap, Type, Union, get_type_hints

from typing_extensions import get_args, get_origin

log = logging.getLogger(__name__)
strict: bool = False


def all_annotations(cls) -> ChainMap[str, Type]:
    """
    Returns a dictionary-like ChainMap that includes annotations for all
    attributes defined in cls or inherited from superclasses.
    Also resolve runtime type hints - https://peps.python.org/pep-0563/
    """
    try:
        return ChainMap(*(get_type_hints(c) for c in cls.__mro__))
    except Exception:
        log.exception(f"Could not get annotations of {cls.__name__}")
        raise


def _init_value(classname: str, dstname: str, dsttype: Any, srcval: Any):
    def msg(prefix: str = "Error when ") -> str:
        return (
            f"{prefix}constructing class {classname!r} expected type {dsttype!r} with origin {dstorigin!r}"
            f" for field {dstname!r}, but received {type(srcval)} with value: {srcval!r}"
        )

    dstorigin = get_origin(dsttype)
    # print(msg("DATADICT INFO "))
    try:
        if dstorigin is list:
            assert type(srcval) is dstorigin, msg()
            return [
                _init_value(classname, dstname, get_args(dsttype)[0], x) for x in srcval
            ]
        elif dstorigin is dict:
            assert type(srcval) is dstorigin, msg()
            return {
                _init_value(classname, dstname, get_args(dsttype)[0], k): _init_value(
                    classname, dstname, get_args(dsttype)[1], v
                )
                for k, v in srcval.items()
            }
        elif dstorigin == Union:
            if type(srcval) in dsttype.__args__:
                return srcval
            return _init_value(classname, dstname, get_args(dsttype)[0], srcval)
        elif dsttype == Any:
            return srcval
        elif issubclass(dsttype, DataDict):
            return dsttype(srcval)
        elif issubclass(dsttype, enum.Enum):
            return dsttype(srcval)
        # elif callable(dsttype):
        #     return dsttype(srcval)
    except Exception:
        log.exception(f"DATADICT:ERROR: {msg()}")
        if strict:
            raise
    return srcval


def _asdict_value(fname: str, val: Any):
    if isinstance(val, list):
        return [_asdict_value(fname, x) for x in val]
    elif isinstance(val, dict):
        return {
            _asdict_value(fname, k): _asdict_value(fname, v) for k, v in val.items()
        }
    elif isinstance(val, enum.Enum):
        return val.value
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
                try:
                    self.__dict__[akey] = copy.deepcopy(self.__class__.__dict__[akey])
                except TypeError:
                    log.exception(
                        f"DATADICT:ERROR: Error when constructing {self.__class__.__name__}: cannot deepcopy {akey} default value"
                    )
                    raise
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
        """Convert datadict to a dictionary"""
        fname = self.asdict.__name__
        return {k: _asdict_value(fname, v) for k, v in self.__dict__.items()}

    def update(self, o: Union[DataDict, dict]):
        """Update values with other values"""
        if isinstance(o, DataDict):
            self.__dict__.update(o)
        else:
            self.__dict__.update(self.__class__(o))
        return self

    def remove_none(self, recursive=True):
        """Remove all elements that are set to None"""
        self.__dict__ = {
            key: val.remove_none() if recursive and isinstance(val, DataDict) else val
            for key, val in self.__dict__.items()
            if val is not None
        }
        return self

    def __repr__(self):
        data = " ".join(f"{k}={self.__dict__[k]!r}" for k in sorted(self.__dict__))
        return f"{self.__class__.__name__}({data})"

    def __eq__(self, o):
        if self.__class__ == o.__class__:
            return self.__dict__ == o.__dict__
        else:
            return False
