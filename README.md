# nomad-tools

Set of tools and utilities to ease interacting with Hashicorp Nomad scheduling solution.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [nomad-watch](#nomad-watch)
  - [nomad-vardir](#nomad-vardir)
  - [nomad-cp](#nomad-cp)
  - [nomad-gitlab-runner](#nomad-gitlab-runner)
  - [import nomad_tools](#import-nomad_tools)
- [Contributing](#contributing)
- [License](#license)

# Installation

```
pipx install nomad-tools
```

# Usage

There are the following command line tools installed as part of this package:

## nomad-watch

Nomad watch watches over a job or allocation, writes the allocation messages, writes the allocation logs, exits with the job exit status. There are multiple modes of operation of the command.

I primarily use nomad-watch for debug watching a running job. Imagine you have a job and the job is failing for an unknown reason. On one terminal you run `nomad-watch -f job <thejob>` and you get up to date stream of logs of the job. I use this in day to day operation and debugging.

Another one is `nomad-watch run job.nomad.hcl`. This is used to start, run, get logs, and get exit status of some one-off job. This can be used to run batch job one-shot processing jobs and stream the logs to current terminal.

Internally, it uses Nomad event stream to get the events in real time.



```
+ nomad-watch --help
Usage: nomad-watch [OPTIONS] COMMAND [ARGS]...

  Run a Nomad job in Nomad and then print logs to stdout and wait for the job to
  be completely finish. Made for running batch commands and monitoring them
  until they are done.

  If the option --no-preserve-exit is given, then exit with the following status:
      0    if operation was successful - the job was run or was purged on --purge
  Ohterwise, when mode is alloc, run, job, stop or stopped, exit with the following status:
      ?    when the job has one task, with that task exit status
      0    if all tasks of the job exited with 0 exit status
      124  if any of the job tasks have failed
      125  if all job tasks have failed
      126  if any tasks are still running
      127  if job has no started tasks
  When the mode is start or started, then exit with the following status:
      0    all tasks of the job have started running
  In either case, exit with the following status:
      1    if some error occured, like python exception

  Examples:
      nomad-watch --namespace default run ./some-job.nomad.hcl
      nomad-watch job some-job
      nomad-watch alloc af94b2
      nomad-watch -N services --task redis -1f job redis

Options:
  -N, --namespace TEXT            Finds Nomad namespace matching given prefix
                                  and sets NOMAD_NAMESPACE environment variable.
                                  [default: default]
  -a, --all                       Do not exit after the current job version is
                                  finished. Instead, watch endlessly for any
                                  existing and new allocations of a job.
  -s, --stream [all|alloc|a|stdout|out|o|1|stderr|err|e|2]
                                  Print only messages from allocation and stdout
                                  or stderr of the task. This option is
                                  cummulative.
  -v, --verbose                   Be verbose
  --json                          job input is in json form, passed to nomad
                                  command with -json
  --stop                          Only relevant in run mode. Stop the job before
                                  exiting.
  --purge-successful              Only relevant in run and stop modes. Purge the
                                  job only if all job tasks finished
                                  successfully.
  --purge                         Only relevant in run and stop modes. Purge the
                                  job.
  -n, --lines INTEGER             Sets the tail location in best-efforted number
                                  of lines relative to the end of logs. Default
                                  prints all the logs. Set to 0 to try try best-
                                  efforted logs from the current log position.
                                  See also --lines-timeout.  [default: -1]
  --lines-timeout FLOAT           When using --lines the number of lines is
                                  best-efforted by ignoring lines for specific
                                  time  [default: 0.5]
  --shutdown-timeout FLOAT        Rather leave at 2 if you want all the logs.
                                  [default: 2]
  -f, --follow                    Shorthand for --all --lines=10 to act similar
                                  to tail -f.
  --no-follow                     Just run once, get the logs in a best-effort
                                  style and exit.
  -t, --task COMPILE              Only watch tasks names matching this regex.
  --polling                       Instead of listening to Nomad event stream,
                                  periodically poll for events
  -x, --no-preserve-status        Do not preserve tasks exit statuses
  -T, --log-timestamp             Additionally add timestamp of the logs from
                                  the task. The timestamp is when the log was
                                  received. Nomad does not store timestamp of
                                  logs sadly.
  --log-timestamp-format TEXT     [default: %Y-%m-%dT%H:%M:%S%z]
  --log-format-alloc TEXT         [default:
                                  %(cyan)s%(allocid).6s:%(group)s:%(task)s:A
                                  %(asctime)s %(message)s%(reset)s]
  --log-format-stderr TEXT        [default:
                                  %(orange)s%(allocid).6s:%(group)s:%(task)s:E
                                  %(message)s%(reset)s]
  --log-format-stdout TEXT        [default: %(allocid).6s:%(group)s:%(task)s:O
                                  %(message)s]
  --log-long-alloc                Log full length allocation id
  -G, --log-no-group              Do not log group
  --log-no-task                   Do not log task
  -1, --log-only-task             Prefix the lines only with task name.
  -0, --log-none                  Log only stream prefix
  -h, --help                      Show this message and exit.
  --version

Commands:
  alloc    Watch over specific allocation
  job      Watch a Nomad job, show its logs and events.
  run      Run a Nomad job and then watch over it until it is finished.
  start    Start a Nomad Job.
  started  Watch a Nomad job until the job has all allocations running.
  stop     Stop a Nomad job and then watch the job until it is stopped -...
  stopped  Watch a Nomad job until the job is stopped - has not running...

  Written by Kamil Cukrowski 2023. Licensed under GNU GPL version 3 or later.
  Licenses.

```



```
+ nomad-watch alloc --help
Usage: nomad-watch alloc [OPTIONS] ALLOCID

  Watch over specific allocation

Options:
  -h, --help  Show this message and exit.
  --version

```



```
+ nomad-watch run --help
Usage: nomad-watch run [OPTIONS] JOBFILE

  Run a Nomad job and then watch over it until it is finished.

Options:
  -h, --help  Show this message and exit.
  --version

```



```
+ nomad-watch job --help
Usage: nomad-watch job [OPTIONS] JOBID

  Watch a Nomad job, show its logs and events.

Options:
  -h, --help  Show this message and exit.
  --version

```



```
+ nomad-watch start --help
Usage: nomad-watch start [OPTIONS] JOBFILE

  Start a Nomad Job. Then act like started mode.

Options:
  -h, --help  Show this message and exit.
  --version

```



```
+ nomad-watch started --help
Usage: nomad-watch started [OPTIONS] JOBID

  Watch a Nomad job until the job has all allocations running. Exit with 2 exit
  status when the job has status dead.

Options:
  -h, --help  Show this message and exit.
  --version

```



```
+ nomad-watch stop --help
Usage: nomad-watch stop [OPTIONS] JOBID

  Stop a Nomad job and then watch the job until it is stopped - has no running
  allocations.

Options:
  -h, --help  Show this message and exit.
  --version

```



```
+ nomad-watch stopped --help
Usage: nomad-watch stopped [OPTIONS] JOBID

  Watch a Nomad job until the job is stopped - has not running allocation.

Options:
  -h, --help  Show this message and exit.
  --version

```



## nomad-vardir

I was severely many times frustrated from the `template` in Nomad jobs. Posting a new job with new `template` content _restarts_ the job. Always, there is nothing you can do about it.

Actually there is. You can set the template to be `data = "{{with nomadVar \"nomad/jobs/redis\"}}{{index . \"./dir/config_file\"}}{{end}}"`. Then, you can create a directory `dir` with a configuration file `config_file`. Using `nomad-vardir --job redis put ./dir` will upload the `config_file` into the JSON in Nomad variables store in the path `nomad/jobs/redis`. Then, you can run a job. Then, when you change configuration file, you would execute `nomad-vardir` again to refresh the content of Nomad variables. Nomad will catch that variables changed and refresh the templates, _but_ it will not restart the job. Instead the change_mode action in template configuration will be executed, which can be a custom script.



```
+ nomad-vardir --help
Usage: nomad-vardir [OPTIONS] COMMAND [ARGS]...

  Given a list of files puts the file content into a nomad variable storage.

Options:
  -n, --dryrun
  -v, --verbose
  -N, --namespace TEXT  Finds Nomad namespace matching given prefix and sets
                        NOMAD_NAMESPACE environment variable.  [default:
                        default]
  -p, --path TEXT       The path of the variable to save
  -j, --job TEXT        Equal to --path=nomad/job/<JOB>
  -s, --service FILE    Get namespace and job name from this nomad service file
  --disable-size-check  Disable checking if the file is smaller than 10mb
  -h, --help            Show this message and exit.
  --version

Commands:
  diff  Show only diff
  get   Get files stored in nomad variables adnd store them in specific...
  put   Recursively scan files in given PATHS and upload filenames as key...

  Written by Kamil Cukrowski 2023. All right reserved.

```



```
+ nomad-vardir put --help
Usage: nomad-vardir put [OPTIONS] [PATHS]...

  Recursively scan files in given PATHS and upload filenames as key and file
  content as value to nomad variable store.

Options:
  --force                Like nomad var put -force
  --check-index INTEGER  Like nomad var put -check-index
  --relative DIRECTORY   Files have paths relative to this directory instead of
                         current working directory
  -D TEXT                Additional var=value to store in nomad variables
  --clear                Remove keys that are not found in files
  -h, --help             Show this message and exit.
  --version

```



```
+ nomad-vardir diff --help
Usage: nomad-vardir diff [OPTIONS] [PATHS]...

  Show only diff

Options:
  --relative DIRECTORY  Files have paths relative to this directory instead of
                        current working directory
  -h, --help            Show this message and exit.
  --version

```



```
+ nomad-vardir get --help
Usage: nomad-vardir get [OPTIONS] DEST

  Get files stored in nomad variables adnd store them in specific directory

Options:
  -h, --help  Show this message and exit.
  --version

```



## nomad-cp

This is a copy of the `docker cp` command. The syntax is the same and similar. However, Nomad does not have the capability of accessing any file inside the allocation filesystem. Instead, `nomad-cp` executes several `nomad exec` calls to execute a `tar` pipe to stream the data from or to the allocation context to or from the local host using stdout and stdin forwarded by `nomad exec`. This is not perfect, and it's API may change in the future.




```
+ nomad-cp --help
Usage: nomad-cp [OPTIONS] SOURCE DEST

  Copy files/folders between a nomad allocation and the local filesystem. Use
  '-' as the source to read a tar archive from stdin and extract it to a
  directory destination in a container. Use '-' as the destination to stream a
  tar archive of a container source to stdout.

  Both source and dest take one of the forms:
      ALLOCATION:SRC_PATH
      JOB:SRC_PATH
      JOB@TASK:SRC_PATH
      SRC_PATH
      -

  Examples:
    {log.name} -n 9190d781:/tmp ~/tmp
    {log.name} -vn -Nservices -job promtail:/. ~/tmp

Options:
  -n, --dry-run                   Do tar -vt for unpacking. Usefull for listing
                                  files for debugging.
  -v, --verbose
  -N, -namespace, --namespace TEXT
                                  Nomad namespace
  -a, --archive                   Archive mode (copy all uid/gid information)
  -j, -job, --job                 Use a **random** allocation from the specified
                                  job ID.
  --test                          Run tests
  -h, --help                      Show this message and exit.
  --version

  Written by Kamil Cukrowski 2023. Licensed under GNU GPL version 3 or later.
  License.

```



TODO:
- Better job searching function.
- Allow specifying task within a job by name instead of allocation name. Refactor options.

### nomad-gitlab-runner

Custom gitlab executor driver on Nomad.



```
+ nomad-gitlab-runner --help
Usage: nomad-gitlab-runner [OPTIONS] COMMAND [ARGS]...

  This is a script implemeting custom gitlab-runner executor to run jobs in
  Nomad job from custom gitlab executor.

  The /etc/gitlab-runner/config.yaml configuration file should look like:
    [[runners]]
    id = 27898742
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

  Example /etc/gitlab-runner/nomad-gitlab-runner.yaml configuration file:
      ---
      default:
          # You can use NOMAD_* variables here
          NOMAD_TOKEN: "1234567"
          NOMAD_ADDR: "http://127.0.0.1:4646"
      # Id of the runner from config.yaml file allows overriding the values for specific runner.
      27898742:
          # Mode to use - "raw_exec", "exec", "docker" or "custom"
          mode: "docker"
          purge: false
          verbose: 0
          CPU: 2048
          MemoryMB: 2048
          docker:
              image: "alpine"
              privileged: false
              services:
                  privileged: true
          # If it possible to override some things.
          override:
              task_config:
                  cpuset_cpus: "1-3"

  Example .gitlab-ci.yml with dockerd service:
      ---
      docker_dind_tls:
          image: docker:24.0.5
          services:
              - docker:24.0.5-dind
          variables:
              DOCKER_HOST: tcp://docker:2376
              DOCKER_TLS_CERTDIR: "/alloc"
              DOCKER_TLS_VERIFY: 1
          script;
              - docker info
      docker_dind_notls:
          image: docker:24.0.5
          services:
              - docker:24.0.5-dind
          variables:
              DOCKER_HOST: tcp://docker:2375
          script;
              - docker info



Options:
  -v, --verbose
  -c, --config FILE   Path to configuration file.  [default: /etc/gitlab-
                      runner/nomad-gitlab-runner.yaml]
  -s, --section TEXT  An additional section read from configuration file to
                      merge with defaults. The value defaults to
                      CUSTOM_ENV_CI_RUNNER_ID which is set to the unique ID of
                      the runner being used.
  -h, --help          Show this message and exit.
  --version

Commands:
  cleanup     https://docs.gitlab.com/runner/executors/custom.html#cleanup
  config      https://docs.gitlab.com/runner/executors/custom.html#config
  prepare     https://docs.gitlab.com/runner/executors/custom.html#prepare
  run         https://docs.gitlab.com/runner/executors/custom.html#run
  showconfig  Show current configuration

  Written by Kamil Cukrowski 2023. Licensed under GNU GPL version 3 or later.
  License.

```



```
+ nomad-gitlab-runner config --help
Usage: nomad-gitlab-runner config [OPTIONS]

  https://docs.gitlab.com/runner/executors/custom.html#config

Options:
  --help  Show this message and exit.

```



```
+ nomad-gitlab-runner prepare --help
Usage: nomad-gitlab-runner prepare [OPTIONS]

  https://docs.gitlab.com/runner/executors/custom.html#prepare

Options:
  --help  Show this message and exit.

```



```
+ nomad-gitlab-runner run --help
Usage: nomad-gitlab-runner run [OPTIONS] SCRIPT STAGE

  https://docs.gitlab.com/runner/executors/custom.html#run

Options:
  --help  Show this message and exit.

```



```
+ nomad-gitlab-runner cleanup --help
Usage: nomad-gitlab-runner cleanup [OPTIONS]

  https://docs.gitlab.com/runner/executors/custom.html#cleanup

Options:
  --help  Show this message and exit.

```



```
+ nomad-gitlab-runner showconfig --help
Usage: nomad-gitlab-runner showconfig [OPTIONS]

  Show current configuration

Options:
  --help  Show this message and exit.

```



### import nomad_tools

The internal API is not at all stable and is an implementation detail as of now. `nomadlib` is internal python library that implements wrappers around some Nomad API responses and a connection to Nomad using requests. It automatically transfers from dictionary into Python object fields and back using a custom wrapper object. The API is extensible and is not full, just basic fields. Pull requests are welcome.

TODO:
- add more fields
- add enums from structs.go

# Contributing

Kindly make a issue or pull request on github. I should be fast to respond and contributions are super welcome.

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

Run that docker with an interactive shell:

```
make docker_shell
```

# License


