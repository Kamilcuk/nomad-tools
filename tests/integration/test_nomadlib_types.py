from nomad_tools import nomadlib
from nomad_tools.common_nomad import mynomad


def test_nomadlib_types_alloc():
    [nomadlib.Alloc(x) for x in mynomad.get("allocations")]


def test_nomadlib_types_jobs():
    [nomadlib.JobsJob(x) for x in mynomad.get("jobs")]
