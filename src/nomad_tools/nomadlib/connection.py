import base64
import dataclasses
import logging
import os
import ssl
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests.adapters
import requests.auth
import websocket

from ..common_base import cached_property
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


class APIException(requests.HTTPError):
    def __init__(self, e: requests.HTTPError):
        """Just construct from other requests.HTTPError"""
        super().__init__(request=e.request, response=e.response)


class PermissionDenied(APIException):
    pass


class JobNotFound(APIException):
    pass


class VariableNotFound(APIException):
    pass


class LogNotFound(APIException):
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
        return self.request("GET", *args, stream=True, **kvargs)


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
                params={cas: cas} if cas is not None else None,
            )
        except requests.HTTPError as e:
            if e.response and e.response.status_code == 409:
                raise VariableConflict(types.Variable(e.response.json()))
            raise

    def delete(self, var_path: str, cas: Optional[int] = None):
        # request does not read DELETE response, so there is no JSON. Call requests, instead of wrapper above.
        self.r.request(
            "DELETE", var_path, params={cas: cas} if cas is not None else None
        )


class NomadConn(Requestor):
    """Represents connection to Nomad"""

    def __init__(self, namespace: str = "", session: Optional[requests.Session] = None):
        self.namespace = namespace
        self.session: requests.Session = session or _default_session()
        self.variables = VariableConn(self, "var")

    @cached_property
    def nomad_version(self) -> str:
        agent = self.get("agent/self")
        version = agent["config"]["Version"]
        if isinstance(version, str):
            return version
        return version["Version"]

    @staticmethod
    def addr() -> str:
        return os.environ.get("NOMAD_ADDR", "http://127.0.0.1:4646")

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
        req = self.session.request(
            method,
            self.addr() + "/v1/" + url,
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
            req.raise_for_status()
        except requests.HTTPError as e:
            code = req.status_code
            text = req.text.lower()
            url = req.url
            if code == 500 and text == "permission denied":
                raise PermissionDenied(e) from e
            elif code == 404 and text == "job not found":
                raise JobNotFound(e) from e
            elif "/v1/var/" in url and code == 404 and text == "variable not found":
                raise VariableNotFound(e) from e
            elif (
                "/v1/client/fs/logs/" in url
                and code in (404, 500)
                and "no such file or directory" in text
            ):
                raise LogNotFound(e) from e
            else:
                log.exception(f"{code} {text!r}")
            raise
        return req

    def jobhcl2json(self, hcl: str):
        return self.post("jobs/parse", json={"JobHCL": hcl})

    def start_job(self, jobjson: dict, submission: Optional[str] = None):
        return self.post("jobs", json={"Job": jobjson, "Submission": submission})

    def stop_job(self, jobid: str, purge: bool = False):
        assert self.namespace
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


def create_websocket_connection(path: str) -> websocket.WebSocket:
    # Replace http in address to ws
    addr: str = NomadConn.addr()
    if addr.startswith("http"):
        addr = "ws" + addr[4:]
    url: str = f"{addr}/{path}"
    # Build headers with authorization and token.
    headers: Dict[str, str] = {}
    token = os.environ.get("NOMAD_TOKEN")
    if token:
        headers["X-Nomad-Token"] = token
    auth = os.environ.get("NOMAD_HTTP_AUTH")
    if auth:
        headers["Authorization"] = "Basic " + base64.b64encode(auth.encode()).decode()
    # Build SSL options from environment variables.
    sslopt: Dict[str, Any] = {}
    skip_verify = os.environ.get("NOMAD_SKIP_VERIFY")
    if skip_verify:
        sslopt["cert_reqs"] = ssl.CERT_NONE
        sslopt["check_hostname"] = False
    cacert = os.environ.get("NOMAD_CACERT")
    if cacert:
        sslopt["ca_cert_path"] = cacert
    cert = os.environ.get("NOMAD_CLIENT_CERT")
    key = os.environ.get("NOMAD_CLIENT_KEY")
    if cert and key:
        sslopt["certfile"] = cert
        sslopt["keyfile"] = key
    # Make the connection.
    return websocket.create_connection(url, header=headers, sslopt=sslopt)
