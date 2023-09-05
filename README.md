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
  -a, --all                       Do not exit after the current job monitoring
                                  is done. Instead, watch endlessly for any
                                  existing and new allocations of a job.
  -s, --stream [all|alloc|stdout|stderr|out|err|1|2]
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
  -f, --follow                    Shorthand for --all --lines=10 to be like in
                                  tail -f.
  --no-follow                     Just run once, get the logs in a best-effort
                                  style and exit.
  -t, --task COMPILE              Only watch tasks names matching this regex.
  -h, --help                      Show this message and exit.

Commands:
  alloc     Watch over specific allocation
  job       Watch over a specific job.
  run       Will run specified Nomad job and then watch over it.
  start     Start a Nomad Job and monitor it until all allocations are...
  starting  Wait until the job has started
  stop      Stop a Nomad job and then wait until it is dead
  stopping  Monitor a Nomad job until it is dead
  test

  Written by Kamil Cukrowski 2023. Jointly under Beerware and MIT Licenses.
```

# Contributing

Make a issue or pull request on github.

# License


