import dataclasses
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional

import requests.adapters
import requests.auth

from . import types

log = logging.getLogger(__name__)


def _default_session():
    s = requests.Session()
    # Increase the number of connections.
    a = requests.adapters.HTTPAdapter(
        pool_connections=1000,
        pool_maxsize=1000,
        max_retries=requests.adapters.Retry(3),
    )
    s.mount("http://", a)
    s.mount("https://", a)
    return s


class PermissionDenied(Exception):
    pass


class JobNotFound(Exception):
    pass


class VariableNotFound(Exception):
    pass


@dataclasses.dataclass
class VariableConflict(Exception):
    variable: types.Variable


class Requestor(ABC):
    @abstractmethod
    def request(self, method: str, url: str, *args, **kvargs) -> requests.Response:
        raise NotImplementedError()

    def _reqjson(self, mode: str, *args, **kvargs):
        rr = self.request(mode, *args, **kvargs)
        return rr.json()

    def get(self, *args, **kvargs):
        return self._reqjson("GET", *args, **kvargs)

    def put(self, *args, **kvargs):
        return self._reqjson("PUT", *args, **kvargs)

    def post(self, *args, **kvargs):
        return self._reqjson("POST", *args, **kvargs)

    def delete(self, *args, **kvargs):
        return self._reqjson("DELETE", *args, **kvargs)

    def stream(self, *args, **kvargs):
        stream = self.request("GET", *args, stream=True, **kvargs)
        return stream


@dataclasses.dataclass
class ChildRequestor(Requestor):
    parent: "Requestor"
    path: str

    def request(self, method: str, url: str, *args, **kvargs):
        return self.parent.request(method, self.path + "/" + url, *args, **kvargs)


class _Conn:
    def __init__(self, parent: "Requestor", path: str):
        self.r = ChildRequestor(parent, path)


class VariableConn(_Conn):
    def read(self, var_path: str) -> types.Variable:
        return types.Variable(self.r.get(var_path))

    def create(
        self, var_path: str, items: Dict[str, str], cas: Optional[int] = None
    ) -> types.Variable:
        try:
            return types.Variable(
                self.r.put(var_path, json={"Items": items}),
                params={cas: cas} if cas else None,
            )
        except requests.HTTPError as e:
            if e.response.status_code == 409:
                raise VariableConflict(types.Variable(e.response.json()))
            raise e

    def delete(self, var_path: str):
        return self.r.delete(var_path)


class NomadConn(Requestor):
    """Represents connection to Nomad"""

    def __init__(self, namespace: str = "", session: Optional[requests.Session] = None):
        self.namespace = namespace
        self.session: requests.Session = session or _default_session()
        self.variables = VariableConn(self, "var")

    def request(
        self,
        method: str,
        url: str,
        params: Optional[dict] = None,
        *args,
        **kvargs,
    ):
        params = dict(params or {})
        params.setdefault(
            "namespace", self.namespace or os.environ.get("NOMAD_NAMESPACE", "*")
        )
        ret = self.session.request(
            method,
            os.environ.get("NOMAD_ADDR", "http://127.0.0.1:4646") + "/v1/" + url,
            *args,
            auth=(
                requests.auth.HTTPBasicAuth(*os.environ["NOMAD_HTTP_AUTH"].split(":"))
                if "NOMAD_HTTP_AUTH" in os.environ
                else None
            ),
            headers=(
                {"X-Nomad-Token": os.environ["NOMAD_TOKEN"]}
                if "NOMAD_TOKEN" in os.environ
                else None
            ),
            params=params,
            verify=False
            if "NOMAD_SKIP_VERIFY" in os.environ
            else os.environ.get("NOMAD_CACERT"),
            cert=(
                (os.environ["NOMAD_CLIENT_CERT"], os.environ["NOMAD_CLIENT_KEY"])
                if "NOMAD_CLIENT_CERT" in os.environ
                and "NOMAD_CLIENT_KEY" in os.environ
                else None
            ),
            **kvargs,
        )
        try:
            ret.raise_for_status()
        except requests.HTTPError as e:
            resp = (ret.status_code, ret.text.lower())
            if resp == (500, "permission denied"):
                raise PermissionDenied(str(e)) from e
            elif resp == (404, "job not found"):
                raise JobNotFound(str(e)) from e
            elif resp == (404, "variable not found"):
                raise VariableNotFound(str(e)) from e
            elif ret.status_code == 500:
                log.exception(resp)
            raise
        return ret

    def jobhcl2json(self, hcl: str):
        return self.post("jobs/parse", json={"JobHCL": hcl})

    def start_job(self, jobjson: dict, submission: Optional[str] = None):
        return self.post("jobs", json={"Job": jobjson, "Submission": submission})

    def stop_job(self, jobid: str, purge: bool = False):
        assert self.namespace
        if purge:
            log.info(f"Purging job {jobid}")
        else:
            log.info(f"Stopping job {jobid}")
        resp: dict = self.delete(f"job/{jobid}", params={"purge": purge})
        assert resp["EvalID"], f"Stopping {jobid} did not trigger evaluation: {resp}"
        return resp

    def find_last_not_stopped_job(self, jobid: str) -> dict:
        assert self.namespace
        jobinit = self.get(f"job/{jobid}")
        if jobinit["Stop"]:
            # Find last job version that is not stopped.
            versions = self.get(f"job/{jobid}/versions")
            notstopedjobs = [job for job in versions["Versions"] if not job["Stop"]]
            if notstopedjobs:
                notstopedjobs.sort(key=lambda job: -job["ModifyIndex"])
                return notstopedjobs[0]
        return jobinit
