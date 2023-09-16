import json
from typing import Any, Dict


def _parse(name: str, data: Dict[Any, Any]):
    classes = []
    lines = []
    for key, val in data.items():
        line = ""
        if isinstance(val, dict):
            classes += _parse(f"{name}{key}", val)
            line = f"{key}: {name}{key}"
        elif isinstance(val, list):
            val = val[0]
            if any(isinstance(val, x) for x in [str, int]):
                line = f"{key}: List[{val.__class__.__name__}]"
            elif isinstance(val, dict):
                classes += _parse(f"{name}{key}", val)
                line = f"{key}: List[{name}{key}]"
        elif any(isinstance(val, x) for x in [str, int]):
            line = f"{key}: {val.__class__.__name__}"
        elif val is None:
            line = f"{key}: Optional[Any] = None"
        assert line, f"{key} {val} {type(val)}"
        lines += [f"  {line}"]
    if name:
        cl = "\n".join([f"class {name}:"] + lines)
        classes += [cl]
    return classes


def generate(name: str, data: Dict[Any, Any]):
    return "\n\n".join(_parse(name, data))


def main(name: str, txt: str):
    print(generate(name, json.loads(txt)))
    print()
    print()


if __name__ == "__main__":
    print(generate("Test", {"a": 1, "b": "str", "c": {"d": "str", "e": [2, 2, 3]}}))

    main(
        "Job",
        """
        {
    "Region": null,
    "Namespace": null,
    "ID": "example",
    "Name": "example",
    "Type": "service",
    "Priority": null,
    "AllAtOnce": null,
    "Datacenters": [
      "dc1"
    ],
    "NodePool": "prod",
    "Constraints": null,
    "Affinities": null,
    "TaskGroups": [
      {
        "Name": "cache",
        "Count": 1,
        "Constraints": null,
        "Affinities": null,
        "Tasks": [
          {
            "Name": "redis",
            "Driver": "docker",
            "User": "",
            "Lifecycle": null,
            "Config": {
              "auth_soft_fail": true,
              "image": "redis:7",
              "ports": [
                "db"
              ]
            },
            "Constraints": null,
            "Affinities": null,
            "Env": null,
            "Services": null,
            "Resources": {
              "CPU": 500,
              "Cores": null,
              "MemoryMB": 256,
              "MemoryMaxMB": null,
              "DiskMB": null,
              "Networks": null,
              "Devices": null,
              "IOPS": null
            },
            "RestartPolicy": null,
            "Meta": null,
            "KillTimeout": null,
            "LogConfig": null,
            "Artifacts": null,
            "Vault": null,
            "Templates": null,
            "DispatchPayload": null,
            "VolumeMounts": null,
            "Leader": false,
            "ShutdownDelay": 0,
            "KillSignal": "",
            "Kind": "",
            "ScalingPolicies": null,
            "Identity": {
              "Env": true,
              "File": true
            }
          }
        ],
        "Spreads": null,
        "Volumes": null,
        "RestartPolicy": {
          "Interval": 1800000000000,
          "Attempts": 2,
          "Delay": 15000000000,
          "Mode": "fail"
        },
        "ReschedulePolicy": null,
        "EphemeralDisk": {
          "Sticky": null,
          "Migrate": null,
          "SizeMB": 300
        },
        "Update": null,
        "Migrate": null,
        "Networks": [
          {
            "Mode": "",
            "Device": "",
            "CIDR": "",
            "IP": "",
            "DNS": null,
            "ReservedPorts": null,
            "DynamicPorts": [
              {
                "Label": "db",
                "Value": 0,
                "To": 6379,
                "HostNetwork": ""
              }
            ],
            "Hostname": "",
            "MBits": null
          }
        ],
        "Meta": null,
        "Services": [
          {
            "Name": "redis-cache",
            "Tags": [
              "global",
              "cache"
            ],
            "CanaryTags": null,
            "EnableTagOverride": false,
            "PortLabel": "db",
            "AddressMode": "",
            "Address": "",
            "Checks": null,
            "CheckRestart": null,
            "Connect": null,
            "Meta": null,
            "CanaryMeta": null,
            "TaggedAddresses": null,
            "TaskName": "",
            "OnUpdate": "",
            "Provider": "nomad"
          }
        ],
        "ShutdownDelay": null,
        "StopAfterClientDisconnect": null,
        "MaxClientDisconnect": null,
        "Scaling": null,
        "Consul": null
      }
    ],
    "Update": {
      "Stagger": null,
      "MaxParallel": 1,
      "HealthCheck": null,
      "MinHealthyTime": 10000000000,
      "HealthyDeadline": 180000000000,
      "ProgressDeadline": 600000000000,
      "Canary": 0,
      "AutoRevert": false,
      "AutoPromote": null
    },
    "Multiregion": null,
    "Spreads": null,
    "Periodic": null,
    "ParameterizedJob": null,
    "Reschedule": null,
    "Migrate": {
      "MaxParallel": 1,
      "HealthCheck": "checks",
      "MinHealthyTime": 10000000000,
      "HealthyDeadline": 300000000000
    },
    "Meta": null,
    "ConsulToken": null,
    "VaultToken": null,
    "Stop": null,
    "ParentID": null,
    "Dispatched": false,
    "DispatchIdempotencyToken": null,
    "Payload": null,
    "ConsulNamespace": null,
    "VaultNamespace": null,
    "NomadTokenID": null,
    "Status": null,
    "StatusDescription": null,
    "Stable": null,
    "Version": null,
    "SubmitTime": null,
    "CreateIndex": null,
    "ModifyIndex": null,
    "JobModifyIndex": null
  }
""",
    )

    main(
        "Allocation",
        """
 {
    "ClientDescription": "Tasks are running",
    "ClientStatus": "running",
    "CreateIndex": 10,
    "CreateTime": 1636017249798459000,
    "DeploymentStatus": {
      "Canary": false,
      "Healthy": true,
      "ModifyIndex": 15,
      "Timestamp": "2021-11-04T10:14:22.054814+01:00"
    },
    "DesiredDescription": "",
    "DesiredStatus": "run",
    "DesiredTransition": {
      "ForceReschedule": null,
      "Migrate": null,
      "Reschedule": null
    },
    "EvalID": "cb20d15d-861f-8d8d-8253-e93932beea2e",
    "FollowupEvalID": "",
    "ID": "5457f16d-0f87-8e6b-5e91-0c7da3a41eb7",
    "JobID": "example",
    "JobType": "service",
    "JobVersion": 0,
    "ModifyIndex": 15,
    "ModifyTime": 1636017262190928000,
    "Name": "example.cache[0]",
    "Namespace": "default",
    "NodeID": "f476d2b4-02dc-c216-d031-273396727347",
    "NodeName": "linux",
    "PreemptedAllocations": null,
    "PreemptedByAllocation": "",
    "RescheduleTracker": null,
    "TaskGroup": "cache",
    "TaskStates": {
      "redis": {
        "Events": [
          {
            "Details": {},
            "DiskLimit": 0,
            "DisplayMessage": "Task received by client",
            "DownloadError": "",
            "DriverError": "",
            "DriverMessage": "",
            "ExitCode": 0,
            "FailedSibling": "",
            "FailsTask": false,
            "GenericSource": "",
            "KillError": "",
            "KillReason": "",
            "KillTimeout": 0,
            "Message": "",
            "RestartReason": "",
            "SetupError": "",
            "Signal": 0,
            "StartDelay": 0,
            "TaskSignal": "",
            "TaskSignalReason": "",
            "Time": 1636017249803624000,
            "Type": "Received",
            "ValidationError": "",
            "VaultError": ""
          },
          {
            "Details": {
              "message": "Building Task Directory"
            },
            "DiskLimit": 0,
            "DisplayMessage": "Building Task Directory",
            "DownloadError": "",
            "DriverError": "",
            "DriverMessage": "",
            "ExitCode": 0,
            "FailedSibling": "",
            "FailsTask": false,
            "GenericSource": "",
            "KillError": "",
            "KillReason": "",
            "KillTimeout": 0,
            "Message": "Building Task Directory",
            "RestartReason": "",
            "SetupError": "",
            "Signal": 0,
            "StartDelay": 0,
            "TaskSignal": "",
            "TaskSignalReason": "",
            "Time": 1636017249805254000,
            "Type": "Task Setup",
            "ValidationError": "",
            "VaultError": ""
          },
          {
            "Details": {},
            "DiskLimit": 0,
            "DisplayMessage": "Task started by client",
            "DownloadError": "",
            "DriverError": "",
            "DriverMessage": "",
            "ExitCode": 0,
            "FailedSibling": "",
            "FailsTask": false,
            "GenericSource": "",
            "KillError": "",
            "KillReason": "",
            "KillTimeout": 0,
            "Message": "",
            "RestartReason": "",
            "SetupError": "",
            "Signal": 0,
            "StartDelay": 0,
            "TaskSignal": "",
            "TaskSignalReason": "",
            "Time": 1636017252049956000,
            "Type": "Started",
            "ValidationError": "",
            "VaultError": ""
          }
        ],
        "Failed": false,
        "FinishedAt": null,
        "LastRestart": null,
        "Restarts": 0,
        "StartedAt": "2021-11-04T09:14:12.04996Z",
        "State": "running",
        "TaskHandle": null
      }
    }
  }
  """,
    )
