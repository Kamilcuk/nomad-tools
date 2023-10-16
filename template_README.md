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

${nomad_watch}

## nomad-port

Smaller wrapper around Nomad API to mimic `docker port` command with some
better templating.

${nomad_port}

## nomad-vardir

I was severely many times frustrated from the `template` in Nomad jobs. Posting a new job with new `template` content _restarts_ the job. Always, there is nothing you can do about it.

Actually there is. You can set the template to be `data = "{{with nomadVar \"nomad/jobs/redis\"}}{{index . \"./dir/config_file\"}}{{end}}"`. Then, you can create a directory `dir` with a configuration file `config_file`. Using `nomad-vardir --job redis put ./dir` will upload the `config_file` into the JSON in Nomad variables store in the path `nomad/jobs/redis`. Then, you can run a job. Then, when you change configuration file, you would execute `nomad-vardir` again to refresh the content of Nomad variables. Nomad will catch that variables changed and refresh the templates, _but_ it will not restart the job. Instead the change_mode action in template configuration will be executed, which can be a custom script.

${nomad_vardir}

## nomad-cp

This is a copy of the `docker cp` command. The syntax is the same and similar. However, Nomad does not have the capability of accessing any file inside the allocation filesystem. Instead, `nomad-cp` executes several `nomad exec` calls to execute a `tar` pipe to stream the data from or to the allocation context to or from the local host using stdout and stdin forwarded by `nomad exec`. This is not perfect, and it's API may change in the future.


${nomad_cp}

TODO:
- Better job searching function.
- Allow specifying task within a job by name instead of allocation name. Refactor options.

### nomad-gitlab-runner

Custom gitlab executor driver on Nomad.

${nomad_gitlab_runner}

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


