import atexit
import base64
import dataclasses
import logging
import os
import ssl
import sys
import urllib.parse
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import requests.adapters
import requests.auth
import urllib3
import websocket
from typing_extensions import Literal

from ..common_base import cached_property
from . import types

log = logging.getLogger(__name__)

NOMAD_ADDR = "NOMAD_ADDR"
NOMAD_NAMESPACE = "NOMAD_NAMESPACE"
NOMAD_TOKEN = "NOMAD_TOKEN"
NOMAD_HTTP_AUTH = "NOMAD_HTTP_AUTH"
NOMAD_CLIENT_CERT = "NOMAD_CLIENT_CERT"
NOMAD_CLIENT_KEY = "NOMAD_CLIENT_KEY"
NOMAD_CACERT = "NOMAD_CACERT"
NOMAD_CAPATH = "NOMAD_CAPATH"
NOMAD_SKIP_VERIFY = "NOMAD_SKIP_VERIFY"
NOMAD_TLS_SERVER_NAME = "NOMAD_TLS_SERVER_NAME"

if NOMAD_SKIP_VERIFY in os.environ:
    urllib3.disable_warnings()


def _default_session():
    session = requests.Session()
    # Override SNI if requested.
    if NOMAD_TLS_SERVER_NAME in os.environ:
        try:
            from requests_toolbelt.adapters.host_header_ssl import HostHeaderSSLAdapter
        except ImportError:
            print(
                "nomadtools: install requests_toolbelt to use NOMAD_TLS_SERVER_NAME",
                file=sys.stderr,
            )
            raise

        obj = HostHeaderSSLAdapter
    else:
        obj = requests.adapters.HTTPAdapter
    # Increase the number of connections.
    adapter = obj(
        pool_connections=1000,
        pool_maxsize=1000,
        max_retries=requests.adapters.Retry(3),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def urlquote(txt: str) -> str:
    return urllib.parse.quote_plus(txt)


class APIException(requests.HTTPError):
    def __init__(self, e: requests.HTTPError, msg: str = ""):
        """Just construct from other requests.HTTPError"""
        super().__init__(msg, request=e.request, response=e.response)


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


@dataclasses.dataclass
class JobSubmission:
    """https://github.com/hashicorp/nomad/blob/42eacc85e2f749651066f03984e3ec4e456596e0/api/jobs.go#L968"""

    Source: str
    Format: Union[str, Literal["hcl1", "hcl2", "json"]]
    VariableFlags: Dict[str, str] = dataclasses.field(default_factory=dict)
    Variables: str = ""

    @staticmethod
    def mk_hcl(hcl: str):
        return JobSubmission(hcl, "hcl2")

class NomadConn(Requestor):
    """Represents connection to Nomad"""

    def __init__(self, namespace: str = "", session: Optional[requests.Session] = None):
        self.namespace = namespace
        self.session: requests.Session = session or _default_session()
        self.variables = VariableConn(self, "var")
        atexit.register(self.session.close)

    @cached_property
    def nomad_version(self) -> str:
        agent = self.get("agent/self")
        version = agent["config"]["Version"]
        if isinstance(version, str):
            return version
        return version["Version"]

    @staticmethod
    def addr() -> str:
        return os.environ.get(NOMAD_ADDR, "http://127.0.0.1:4646")

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
            "namespace", self.namespace or os.environ.get(NOMAD_NAMESPACE, "*")
        )
        req = self.session.request(
            method,
            self.addr() + "/v1/" + url,
            *args,
            auth=(
                requests.auth.HTTPBasicAuth(*os.environ[NOMAD_HTTP_AUTH].split(":", 1))
                if NOMAD_HTTP_AUTH in os.environ
                else None
            ),
            headers={
                **(
                    {"X-Nomad-Token": os.environ[NOMAD_TOKEN]}
                    if NOMAD_TOKEN in os.environ
                    else {}
                ),
                **(
                    {"Host": os.environ[NOMAD_TLS_SERVER_NAME]}
                    if NOMAD_TLS_SERVER_NAME in os.environ
                    else {}
                ),
            },
            params=params,
            verify=(
                False
                if NOMAD_SKIP_VERIFY in os.environ
                else (
                    os.environ[NOMAD_CACERT]
                    if NOMAD_CACERT in os.environ
                    else (
                        os.environ[NOMAD_CAPATH] if NOMAD_CAPATH in os.environ else True
                    )
                )
            ),
            cert=(
                (os.environ[NOMAD_CLIENT_CERT], os.environ[NOMAD_CLIENT_KEY])
                if NOMAD_CLIENT_CERT in os.environ and NOMAD_CLIENT_KEY in os.environ
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
            elif code == 404 and text in ["job not found", "job versions not found"]:
                raise JobNotFound(e) from e
            elif "/v1/var/" in url and code == 404 and text == "variable not found":
                raise VariableNotFound(e) from e
            elif "/v1/client/fs/logs/" in url and code in (404, 500):
                raise LogNotFound(e) from e
            else:
                log.exception(f"{code} {text!r}")
            raise
        return req

    def jobhcl2json(self, hcl: str):
        return self.post("jobs/parse", json={"JobHCL": hcl})

    def start_job(self, jobjson: dict, submission: Optional[JobSubmission] = None):
        data = {"Job": jobjson}
        if submission:
            data["Submission"] = dataclasses.asdict(submission)
        return self.post("jobs", json=data)

    def stop_job(self, jobid: str, purge: bool = False):
        assert self.namespace
        resp: dict = self.delete(f"job/{urlquote(jobid)}", params={"purge": purge})
        assert resp["EvalID"], f"Stopping {jobid} did not trigger evaluation: {resp}"
        return resp

    def find_last_not_stopped_job(self, jobid: str) -> dict:
        assert self.namespace
        jobinit = self.get(f"job/{urlquote(jobid)}")
        if jobinit["Stop"]:
            # Find last job version that is not stopped.
            versions = self.get(f"job/{urlquote(jobid)}/versions")
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
    token = os.environ.get(NOMAD_TOKEN)
    if token:
        headers["X-Nomad-Token"] = token
    auth = os.environ.get(NOMAD_HTTP_AUTH)
    if auth:
        headers["Authorization"] = "Basic " + base64.b64encode(auth.encode()).decode()
    # Build SSL options from environment variables.
    sslopt: Dict[str, Any] = {}
    skip_verify = os.environ.get(NOMAD_SKIP_VERIFY)
    if skip_verify:
        sslopt["cert_reqs"] = ssl.CERT_NONE
        sslopt["check_hostname"] = False
    cacert = os.environ.get(NOMAD_CACERT)
    if cacert:
        sslopt["ca_certs"] = cacert
    else:
        capath = os.environ.get(NOMAD_CAPATH)
        if capath:
            sslopt["ca_cert_path"] = capath
    cert = os.environ.get(NOMAD_CLIENT_CERT)
    key = os.environ.get(NOMAD_CLIENT_KEY)
    if cert and key:
        sslopt["certfile"] = cert
        sslopt["keyfile"] = key
    # Make the connection.
    return websocket.create_connection(url, header=headers, sslopt=sslopt)
