# nomad-tools
Set of tools and utilities to ease interacting with Hashicorp Nomad scheduling solution.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

# Installation

```
pip install nomad-tools
```

# Usage

There are the following command line tools installed as part of this package:

## nomad-watch

Nomad watch watches over a job or allocation, writes the allocation messages, writes the allocation logs, exits with the job exit status. There are multiple modes of operation of the command.

I primarily use nomad-watch for debug watching a running job. Imagine you have a job and the job is failing for an unknown reason. On one terminal you run `nomad-watch -f job <thejob>` and you get up to date stream of logs of the job. I use this in day to day operation and debugging.

Another one is `nomad-watch run job.nomad.hcl`. This is used to start, run, get logs, and get exit status of some one-off job. This can be used to run batch job one-shot processing jobs and stream the logs to current terminal.

Internally, it uses Nomad event stream to get the events in real time.

```
Usage: nomad-watch [OPTIONS] COMMAND [ARGS]...

  Run a Nomad job in Nomad and then print logs to stdout and wait for the job
  to finish. Made for running batch commands and monitoring them until they
  are done. The exit code is 0 if the job was properly done and all tasks
  exited with 0 exit code. If the job has only one task, exit with the task
  exit status. Othwerwise, if all tasks exited failed, exit with 3. If there
  are failed tasks, exit with 2. If there was a python exception, standard
  exit code is 1. Examples:

  nomad-watch --namespace default run ./some-job.nomad.hcl

  nomad-watch job some-job

  nomad-watch alloc af94b2

  nomad-watch --all --task redis -N services job redis

Options:
  -N, --namespace TEXT            Finds Nomad namespace matching given prefix
                                  and sets NOMAD_NAMESPACE environment
                                  variable.
  -a, --all                       Do not exit after the current job version is
                                  finished. Instead, watch endlessly for any
                                  existing and new allocations of a job.
  -s, --stream [all|alloc|a|stdout|stderr|out|err|o|e|1|2]
                                  Print only messages from allocation and
                                  stdout or stderr of the task. This option is
                                  cummulative.
  -v, --verbose                   Be verbose
  --json                          job input is in json form, passed to nomad
                                  command with --json
  --stop                          In run mode, make sure to stop the job
                                  before exit.
  --purge                         In run mode, stop and purge the job before
                                  exiting.
  -n, --lines INTEGER             Sets the tail location in best-efforted
                                  number of lines relative to the end of logs
                                  [default: -1]
  --lines-timeout FLOAT           When using --lines the number of lines is
                                  best-efforted by ignoring lines for specific
                                  time  [default: 0.5]
  --shutdown_timeout FLOAT        Rather leave at 2 if you want all the logs.
                                  [default: 2]
  -f, --follow                    Shorthand for --all --lines=10 to act
                                  similar to tail -f.
  --no-follow                     Just run once, get the logs in a best-effort
                                  style and exit.
  -t, --task COMPILE              Only watch tasks names matching this regex.
  --log-format-alloc TEXT         [default:
                                  %(cyan)s%(allocid).6s:%(group)s:%(task)s:A
                                  %(now)s %(message)s%(reset)s]
  --log-format-stderr TEXT        [default:
                                  %(orange)s%(allocid).6s:%(group)s:%(task)s:E
                                  %(message)s%(reset)s]
  --log-format-stdout TEXT        [default: %(allocid).6s:%(group)s:%(task)s:O
                                  %(message)s]
  -l, --log-long                  Log full allocation id
  -S, --log-short                 Make the format short by logging only task
                                  name.
  -h, --help                      Show this message and exit.

Commands:
  alloc    Watch over specific allocation
  job      Watch a Nomad job, show its logs and events.
  run      Run a Nomad job and then watch over it until it is finished.
  start    Start a Nomad Job and watch it until all allocations are running
  started  Watch a Nomad job until the job is started.
  stop     Stop a Nomad job and then watch the job until it is stopped -...
  stopped  Watch a Nomad job until the job is stopped - has not running...

  Written by Kamil Cukrowski 2023. Licensed under GNU GPL version 3 or later.
  Licenses.
```

## nomad-vardir

I was severely many times frustrated from the `template` in Nomad jobs. Posting a new job with new `template` content _restarts_ the job. Always, there is nothing you can do about it.

Actually there is. You can set the template to be `data = "{{with nomadVar \"nomad/jobs/redis\"}}{{index . \"./dir/config_file\"}}{{end}}"`. Then, you can create a directory `dir` with a configuration file `config_file`. Using `nomad-vardir --job redis put ./dir` will upload the `config_file` into the JSON in Nomad variables store in the path `nomad/jobs/redis`. Then, you can run a job. Then, when you change configuration file, you would execute `nomad-vardir` again to refresh the content of Nomad variables. Nomad will catch that variables changed and refresh the templates, _but_ it will not restart the job. Instead the change_mode action in template configuration will be executed, which can be a custom script.

```
usage: nomad-vardir [-h] [-n] [-v] [--namespace NAMESPACE] [--job JOB]
                    [-s SERVICE] [-D D] [--disable-size-check] [--clear CLEAR]
                    {put,diff,get} ...

Given a list of files puts the file content into a nomad variable storage.

positional arguments:
  {put,diff,get}        mode to run
    put                 recursively scan given paths and upload them with filenames as key to nomad variable store
    diff                just like put, but stop after showing diff
    get                 Get files stored in nomad variables adnd store them in specific directory

options:
  -h, --help            show this help message and exit
  -n, --dryrun
  -v, --verbose
  --namespace NAMESPACE
  --job JOB             Job name to upload the variables to
  -s SERVICE, --service SERVICE
                        Get namespace and job name from this nomad service file
  -D D                  Additional var=value to store in nomad variables
  --disable-size-check  Disable checking if the file is smaller than 10mb
  --clear CLEAR         clear keys that are not found in files

Written by Kamil Cukrowski 2023. All right reserved.
```

TODO: Rewrite command line options. Write unit tests

## nomad-cp

This is a copy of the `docker cp` command. The syntax is the same and similar. However, Nomad does not have the capability of accessing any file inside the allocation filesystem. Instead, `nomad-cp` executes several `nomad exec` calls to execute a `tar` pipe to stream the data from or to the allocation context to or from the local host using stdout and stdin forwarded by `nomad exec`. This is not perfect, and it's API may change in the future.

```
usage: nomad-cp [-h] [-n] [-d] [-v] [-N NAMESPACE] [-a] [-j] [--test]
                source dest

Copy files/folders between a nomad allocation and the local filesystem.
Use '-' as the source to read a tar archive from stdin
and extract it to a directory destination in a container.
Use '-' as the destination to stream a tar archive of a
container source to stdout.
    

positional arguments:
  source                ALLOCATION:SRC_PATH|JOB:SRC_PATH|SRC_PATH|-
  dest                  ALLOCATION:DEST_PATH|JOB:DEST_PATH|DEST_PATH|-

options:
  -h, --help            show this help message and exit
  -n, --dry-run         Do tar -vt for unpacking. Usefull for listing files for debugging.
  -d, --debug
  -v, --verbose
  -N NAMESPACE, -namespace NAMESPACE, --namespace NAMESPACE
                        Nomad namespace
  -a, --archive         Archive mode (copy all uid/gid information)
  -j, -job, --job       Use a **random** allocation from the specified job ID.
  --test                Run tests

Examples:
  nomad_cp.py -n 9190d781:/tmp ~/tmp
  nomad_cp.py -vn -Nservices -job promtail:/. ~/tmp

Written by Kamil Cukrowski 2023. Licensed under GNU GPL version or later.
```

TODO:
- Better job searching function.
- Allow specifying task within a job by name instead of allocation name. Refactor options.

### nomad-gitlab-runner

This is a WIP project. It has configuration in `/etc/gitlab-runner/nomad.toml`. The command contains a custom implementation of a Nomad gitlab-runner using custom executor. The skeleton is done, all is left is to add a lot more configuration options and error handling. The raw_exec driver works fine. The basic gitlab-runner/config.toml configuration would look like the following:


```
  executor = "custom"
  [runners.custom]
	  config_exec = "nomad-gitlab-runner"
	  config_args = ["config"]
	  prepare_exec = "nomad-gitlab-runner"
	  prepare_args = ["prepare"]
	  run_exec = "nomad-gitlab-runner"
	  run_args = ["run"]
	  cleanup_exec = "nomad-gitlab-runner"
	  cleanup_args = ["cleanup"]
```

TODO:
- add full configuration
- allow running custom services specified in .gitlab-ci.yml

### `from nomad_tools import nomadlib`

`nomadlib` is a python library that implements wrappers around some Nomad API responses and a connection to Nomad using requests. It automatically transfers from dictionary into Python object fields and back using a custom wrapper object. The API is extensible and is not full, just basic fields. Pull requests are welcome.

TODO:
- add more fields
- add enums from structs.go

# Contributing

Kindly make a issue or pull request on github. I should be fast to respond and contributions are super welcome.

## Local development

Install editable package locally with pytest:

```
pip install -e '.[test]'
```

Run unit tests:

```
make test
```

Run integration test by spawning a temporary docker with a temporary Nomad server running inside:

```
make docker_test
```

# License


