from .connection import *
from .types import *


def escape(txt: str) -> str:
    # Protect against "interpolation expression" by nomad.
    # See https://www.nomadproject.io/docs/job-specification/hcl2/expressions#string-templates
    return txt.replace("%{", "%%{").replace("${", "$${")


def unescape(txt: str) -> str:
    # See nomad_escape
    return txt.replace("%%{", "%{").replace("$${", "${")
