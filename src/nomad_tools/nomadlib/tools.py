import datetime


def escape(txt: str) -> str:
    # Protect against "interpolation expression" by nomad.
    # See https://www.nomadproject.io/docs/job-specification/hcl2/expressions#string-templates
    return txt.replace("%{", "%%{").replace("${", "$${")


def unescape(txt: str) -> str:
    # See nomad_escape
    return txt.replace("%%{", "%{").replace("$${", "${")


def ns2s(ns: int) -> float:
    """Convert nanoseconds to float seconds"""
    return ns / 1000000000


def ns2dt(ns: int) -> datetime.datetime:
    """Convert nanoseconds to datetime"""
    return datetime.datetime.fromtimestamp(ns2s(ns)).astimezone()
