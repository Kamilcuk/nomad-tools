# nomadtools

Set of tools and utilities to ease interacting with HashiCorp Nomad scheduling solution.

## Table of Contents


<!-- mtoc-start -->

* [Installation](#installation)
  * [Shell completion](#shell-completion)
* [Usage](#usage)
  * [`watch` - get all logs and events of a Nomad job in terminal](#watch---get-all-logs-and-events-of-a-nomad-job-in-terminal)
  * [`go` - create, run and watch a Nomad job from command line](#go---create-run-and-watch-a-nomad-job-from-command-line)
  * [`constrainteval` - evaluate constraint in command line](#constrainteval---evaluate-constraint-in-command-line)
  * [`listattributes` - List nodes with given attributes or show all available attributes.](#listattributes---list-nodes-with-given-attributes-or-show-all-available-attributes)
  * [`listnodeattributes` - list all available node attributes](#listnodeattributes---list-all-available-node-attributes)
  * [`port` - list ports allocated by job or allocation](#port---list-ports-allocated-by-job-or-allocation)
  * [`vardir` - manipulate nomad variable keys as files](#vardir---manipulate-nomad-variable-keys-as-files)
  * [`cp` - copy files to/from/between nomad allocations](#cp---copy-files-tofrombetween-nomad-allocations)
  * [`gitlab-runner` - dynamically schedule gitlab CI jobs](#gitlab-runner---dynamically-schedule-gitlab-ci-jobs)
  * [`githubrunner` - dynamically schedule GitHub actions jobs](#githubrunner---dynamically-schedule-github-actions-jobs)
  * [`dockers` - list docker images referenced by Nomad job](#dockers---list-docker-images-referenced-by-nomad-job)
  * [`downloadrelease` - download specific Nomad executable version](#downloadrelease---download-specific-nomad-executable-version)
  * [`task` - find task and allocation and execute action on it](#task---find-task-and-allocation-and-execute-action-on-it)
    * [`task exec` - execute a command inside the allocation](#task-exec---execute-a-command-inside-the-allocation)
    * [`task json` - output found allocations and task names in json form](#task-json---output-found-allocations-and-task-names-in-json-form)
    * [`task path` - output path in the form properly escaped for use with nomadtools cp](#task-path---output-path-in-the-form-properly-escaped-for-use-with-nomadtools-cp)
    * [`task xargs` - output in the form -task <task> <allocid> that is usable with xargs nomad alloc](#task-xargs---output-in-the-form--task-task-allocid-that-is-usable-with-xargs-nomad-alloc)
  * [`nodenametoid` - convert node name to id](#nodenametoid---convert-node-name-to-id)
  * [`info topology` - show some information about Nomad node usage](#info-topology---show-some-information-about-nomad-node-usage)
  * [import nomad_tools](#import-nomad_tools)
* [History](#history)
* [Contributing](#contributing)
  * [Running tests](#running-tests)
* [License](#license)

<!-- mtoc-end -->

If you use this package, wheather you like it or not, would you want something to improve,
or you feel like talking, consider leaving your opinion on github discussions
https://github.com/Kamilcuk/nomad-tools/discussions  .

If there is a feature request of bug, create a github issue
https://github.com/Kamilcuk/nomad-tools/issues .

In the world of burning open source developers your comment
will help brighten or darken my day and also serve to prioritize_is_correct
this tool development.

# Installation

You can use docker image:

```
docker run -t --rm -e NOMAD_ADDR ghcr.io/kamilcuk/nomad-tools:0.4.0 --help
```

Install the project using `pipx` project.

```
pipx install nomad-tools
```

After installation the executable `nomadtools` should be available.

```
nomadtools --help
```

## Shell completion

After installation, see `nomadtools watch --autocomplete-info` for shell
completion installation instruction.

# Usage

This module installs command line tool `nomadtools` with several modes of
operation:

## `watch` - get all logs and events of a Nomad job in terminal

`nomadtools watch` is meant to watch over a job change that you type in
terminal. It prints all relevant messages - messages about allocation,
evaluation, deployment and stdout and stderr logs from all the
processes. Depending on the mode of operation, the tool waits until an
action is finished.

I primarily use `watch` to deploy new versions of services. I was always
frustrated that I start something from terminal and then I have to check the
logs of the service in multiple tabs in the Nomad web interface. For example,
you can use `watch start ./postgres.nomad.hcl` to update PostgreSQL
container and watch it's logs in your terminal.

An example terminal session deploying a HTTP server job with canary and health
check. Note that while the new version is deployed, the old one still prints
the logs.

![gif showing example usage of watch start](./assets/imgs/nomad-watch-start-listen.gif)

Another usage of the job is to run an one-shot batch jobs to do something and
wait until they are finished and collect the exit status and logs, for example
as an airflow or cron job. In this case `run` mode will wait for the job to be
finished. For example `watch --purge run ./compute.nomad.hcl` will run
a calculation job, purge after it is done and exit with calculate job exit
status (if there is one task).

![gif showing example usage of watch run](./assets/imgs/nomad-watch-run-compute.gif)

Internally, watch uses Nomad event stream to get the events in real time.

## `go` - create, run and watch a Nomad job from command line

Mimics operation of `docker run`, it is built on top of `watch` mode to
execute a single Nomad job created dynamically from command line arguments.
It creates a Nomad job specification from command line arguments and then
"watches" over the execution of the job.

```
$ nomadtools go --rm alpine apk add bash
INFO:nomad_tools.nomad_watch:Watching job nomad_tools_go_5305da8f-b376-4c35-9a05-71027aadd587@default until it is finished
Allocation 70c4ac9d-0e03-53d7-6e34-9c86cf8ee768 started on leonidas
Received Task received by client
Task Setup Building Task Directory
Driver Downloading image
Started Task started by client
INFO:nomad_tools.nomad_watch:Job nomad_tools_go_5305da8f-b376-4c35-9a05-71027aadd587#0@default started allocations 70c4ac running group 'nomad_tools_go_5305da8f-b376-4c35-9a05-71027aadd587' with 1 main tasks.
fetch https://dl-cdn.alpinelinux.org/alpine/v3.19/main/x86_64/APKINDEX.tar.gz
fetch https://dl-cdn.alpinelinux.org/alpine/v3.19/community/x86_64/APKINDEX.tar.gz
(1/4) Installing ncurses-terminfo-base (6.4_p20231125-r0)
(2/4) Installing libncursesw (6.4_p20231125-r0)
(3/4) Installing readline (8.2.1-r2)
(4/4) Installing bash (5.2.21-r0)
Terminated Exit Code: 0
Allocation 70c4ac9d-0e03-53d7-6e34-9c86cf8ee768 finished
Executing bash-5.2.21-r0.post-install
Executing busybox-1.36.1-r15.trigger
OK: 10 MiB in 19 packages
INFO:nomad_tools.nomad_watch:Purging job nomad_tools_go_5305da8f-b376-4c35-9a05-71027aadd587
INFO:nomad_tools.nomad_watch:Job nomad_tools_go_5305da8f-b376-4c35-9a05-71027aadd587#0@default purged with no active allocations, evaluations nor deployments. Exiting.
INFO:nomad_tools.nomad_watch:Single task exited with 0 exit status. Exit code is 0.
```

## `constrainteval` - evaluate constraint in command line

Evaluate a constraint and show all nodes that match the constraint. In
addition to the node names, it also shows all attributes referenced while
evaluating the constraint given on command line arguments. Useful for
searching for which hosts contain what value of a attribute.

```
$ nomadtools constrainteval '${attr.cpu.arch}' = amd64
name   attr.cpu.arch
node1  amd64
node2  amd64
```

This mode uses a cache in `~/.cache/nomadtools/nodes.json` for caching all the
attributes of nodes downloaded from Nomad. This is used to speed up. The
program needs to make one query for every single node in Nomad, which for a
lot of nodes is costly.

## `listattributes` - List nodes with given attributes or show all available attributes.

```
$ nomadtools listattributes attr.cpu.arch
name   attr.cpu.arch
node1  amd64
node2  amd64
```

Bash completion works.

## `listnodeattributes` - list all available node attributes

Lists all node attributes or attribute of given nodes.
Additionally with a leading dot lists the whole json return from Nomad API.

This is meant to be used with `grep` and other unix tools for easy parsing.

```
$ nomadtools listnodeattributes node1 | grep datacenter
dc.datacenter                                   dc
```

It uses the same cache and option as `constrainteval` option.

## `port` - list ports allocated by job or allocation

Prints out the ports allocated for a particular Nomad job or
allocation. It is meant to mimic `docker port` command.

```
$ nomadtools port httpd
192.168.0.5:31076
$ nomadtools port -l httpd
192.168.0.5 31076 http httpd.listen[0] d409e855-bf13-a342-fe7a-6fb579d2de85
$ nomadtools port --alloc d409e855
192.168.0.5:31076
```

Further argument allows to filter for port label.

```
$ nomadtools port httpd http
192.168.0.5:31076
```

## `vardir` - manipulate nomad variable keys as files

I was frustrated with how Nomad variables look like. It is really hard to
incrementally modify Nomad variables. The API is at one go. You either update
all variables or nothing. Most often I wanted to update a single key
from a Nomad variable at a time and the variable value was usually a file content.

Example execution of putting a `passwordfile.txt` into `nomad/jobs/nginx`
Nomad variable:

```
$ nomadtools vardir -j nginx put ./passwordfile.txt 
nomad_vardir: Putting var nomad/jobs/nginx@default with keys: passwordfile.txt
$ nomadtools vardir -j nginx cat passwordfile.txt 
secretpassword
$ nomadtools vardir -j nginx ls
nomad_vardir: Listing Nomad variable at nomad/jobs/nginx@default
key              size
passwordfile.txt 15
```

You can then remove the `passwordfile.txt` key from the Nomad variable:

```
$ nomadtools vardir -j nginx rm passwordfile.txt 
nomad_vardir: Removing passwordfile.txt
nomad_vardir: Removing empty var nomad/jobs/nginx@default
$ nomadtools vardir -j nginx ls
nomad_vardir: Nomad variable not found at nomad/jobs/nginx@default
```

## `cp` - copy files to/from/between nomad allocations

This is a copy of the `docker cp` command. The syntax is meant to be the
same with docker. The rules of copying a file vs directory are meant to be
in-line with `docker cp` documentation.

`nomadtools cp` uses some special syntax for specifying from which allocation/task
exactly do you want to copy by using colon `:`. The number of colons in the
arguments determines the format. The colon can be escaped with slash `\` in
the path if needed.

Both `SRC` and `DST` addresses can be specified as follows:

```
# Search a task matching specific URL query:
task://JOB[@NAMESPACE]/PATH[?group=GROUP][&alloc=ALLOC][&task=TASK][&hostname=HOSTNAME][&node=NODE]
# or
:ALLOCATION:PATH                  copy path from this allocation having one job
:ALLOCATION::TASK:PATH            copy path from this task inside allocation
:ALLOCATION:GROUP:TASK:PATH       like above, but filter by group name
JOB:PATH                          copy path from one task inside specified job
JOB::TASK:PATH                    copy path from the task inside this job
JOB:GROUP:TASK:PATH               like above, but filter also by group name
PATH                              copy local path
-                                 copy stdin or stdout TAR stream
```

`cp` depends on `sh` and `tar` command line utility to be available
inside the allocation it is coping to/from. It has to be available there.

Example:

```
$ nomadtools cp -v nginx:/etc/nginx/nginx.conf ./nginx.conf
INFO nomad_cp.py:copy_mode:487: File :d409e855-bf13-a342-fe7a-6fb579d2de85:listen:/etc/nginx/nginx.conf -> ./nginx.conf
$ nomadtools cp -v alpine:/etc/. ./etc/
INFO nomad_cp.py:copy_mode:512: New mkdir :d409e855-bf13-a342-fe7a-6fb579d2de85:listen:/etc/. -> /home/kamil/tmp/etc2/
```

Nomad does not have the capability of accessing any file inside the
allocation file system. Instead, `nomad-cp` executes several `nomad exec`
calls to execute a `tar` pipe to stream the data from or to the allocation
context to or from the local host using stdout and stdin forwarded by
`nomad exec`.

## `gitlab-runner` - dynamically schedule gitlab CI jobs

An implementation of custom Gitlab executor driver that runs Gitlab CI/CD jobs
using Nomad.

This program does _not_ run the `gitlab-runner` itself in Nomad. Rather, the
`gitlab-runner` is running on (any) one host. That `gitlab-runner` will then
schedule Nomad jobs to execute using the script as an executor. These jobs will
execute the CI/CD from Gitlab inside Nomad cluster.

More on it can be read on github wiki
[https://github.com/Kamilcuk/nomad-tools/wiki/gitlab%E2%80%90runner](https://github.com/Kamilcuk/nomad-tools/wiki/gitlab%E2%80%90runner) .

## `githubrunner` - dynamically schedule GitHub actions jobs

Schedule Github actions runners as Nomad job for user repositories.

See documentation in [doc/githubrunner.md](doc/githubrunner.md) .

## `dockers` - list docker images referenced by Nomad job

Lists docker images referenced in Nomad job file or a running Nomad job.

```
$ nomadtools dockers ./httpd.nomad.hcl
busybox:stable
$ nomadtools dockers --job httpd
busybox:stable
```

## `downloadrelease` - download specific Nomad executable version

Program for downloading specific Nomad release binary from their release page.
I use it for testing and checking new Nomad versions.

```
$ nomadtools downloadrelease nomad
INFO:nomad_tools.nomad_downloadrelease:Downloading https://releases.hashicorp.com/nomad/1.7.3/nomad_1.7.3_linux_amd64.zip to nomad
INFO:nomad_tools.nomad_downloadrelease:https://releases.hashicorp.com/nomad/1.7.3/nomad_1.7.3_linux_amd64.zip -> -rwxr-xr-x 105.7MB nomad
$ nomadtools downloadrelease consul
INFO:nomad_tools.nomad_downloadrelease:Downloading https://releases.hashicorp.com/consul/1.9.9/consul_1.9.9_linux_amd64.zip to consul
INFO:nomad_tools.nomad_downloadrelease:https://releases.hashicorp.com/consul/1.9.9/consul_1.9.9_linux_amd64.zip -> -rwxr-xr-x 105.8MB consul
$ nomadtools downloadrelease -p 1.6.3 nomad ./nomad1.6.3
INFO:nomad_tools.nomad_downloadrelease:Downloading https://releases.hashicorp.com/nomad/1.6.3/nomad_1.6.3_linux_amd64.zip to nomad1.6.3
INFO:nomad_tools.nomad_downloadrelease:https://releases.hashicorp.com/nomad/1.6.3/nomad_1.6.3_linux_amd64.zip -> -rwxr-xr-x 101.8MB nomad1.6.3
```

## `task` - find task and allocation and execute action on it

`nomad alloc exec` didn't met my expectations. It is boring and tedious to find an allocation ID
and then copy it in the terminal. There is also no bash completion for task names.

I decided I want to have advanced filtering, where given a job name and node name some tool
automatically finds an allocation and task to copy from.

Given a system job promtail running on all machines, `nomadtools task -j promtail -n host1 exec bash -l`
will drop into a bash shell inside promtail job allocation running on host1.

This tool can be used to get a shell to the chosen allocation.

### `task exec` - execute a command inside the allocation

This internally calls `nomad alloc exec`.

```
$ nomadtools task -j mail exec bash -l
root@main:/#
```

### `task json` - output found allocations and task names in json form

### `task path` - output path in the form properly escaped for use with nomadtools cp

```
$ nomadtools cp "$(nomadtools task -j mail path /etc/fstab)" ./fstab
INFO entry_cp.py:copy_mode:650: File :0c18e9aa-f053-cc3e-6fe3-3d23f159c2e5:mail:/etc/fstab -> ./fstab
37.0  B 0:00:00 [73.3  B/s] [ <=>
```

### `task xargs` - output in the form -task <task> <allocid> that is usable with xargs nomad alloc

```
$ nomadtools task -j mail xargs -0 logs -- -stderr | xargs -0 nomad alloc
$ nomadtools task -j mail xargs logs -- -stderr | xargs nomad alloc
$ nomad alloc logs $(nomadtools task -j mail xargs) -stderr
```

## `nodenametoid` - convert node name to id

```
$ nomadtools nodenametoid node1
3e50c2ef-16bd-0253-6635-1a55c25e74ca
```

## `info topology` - show some information about Nomad node usage

```
$ nomadtools info topology
dc node1 ready 2000B/15988MB 1000MHz/12000MHz 1allocs [mariadb.mariadb[0] mariadb services 2000MB 1000MHz]
```

## import nomad_tools

This project is licensed under GPL. The internal API of this project can be
used, however it is not stable at all and is an implementation detail.

Internally, `nomad_tools.nomadlib` is a Python class definitions which
represents models for Nomad API data documentation.

# History

This module once installed bunch of separate tools, like `nomad-watch` or
`nomad-gitlab-runner`. That became unmaintainable. It is one `nomadtools`
executable with several sub-commands.

# Contributing

Kindly make a issue or pull request on GitHub.
I should be fast to respond and contributions are super welcome.

## Running tests

I want to support Python 3.7 with the project.

To test first install editable package locally with test dependencies:

```
pip install -e '.[test]'
```

You can run unit tests always without any external tools:

```
./unit_tests.sh
```

To run integration tests, you have to be able to connect to Nomad server.

```
./integration_tests.sh
./integration_tests.sh -k nomad_vardir
```

# License

GPL

