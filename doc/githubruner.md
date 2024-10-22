# githubrunner

<!-- vim-markdown-toc GFM -->

* [What it does?](#what-it-does)
* [How it works?](#how-it-works)
* [How start the scheduler?](#how-start-the-scheduler)
* [Nomad job template](#nomad-job-template)
* [How to use it from Github workflow?](#how-to-use-it-from-github-workflow)
* [Example workflow:](#example-workflow)

<!-- vim-markdown-toc -->

# What it does?

Runs a specific Nomad job matching the labels requested by pending GitHub actions jobs.
Controls the number of Nomad jobs depending on the number of pending GithHub actions jobs.

# How it works?

- Every CONFIG.pool seconds
  - Get all actions workflow runs with the following algorithm:
    - For each entry configured CONFIG.repos
      - If the repository has no /
        - Get all repositories under this organization or user
      - else use the repository
    - For each such repository
      - For each repository actions workflow
        - For each repository actions workflow run
          - If the workflow run has status queued or in_progress
            - Add it to the list
  - Get all Nomad jobs with the following algorithm:
    - For all Nomad jobs in CONFIG.nomad.namespace
      - If the job name starts with CONFIG.nomad.jobprefix + "-"
        - If the `job["Meta"][CONFIG.nomad.meta]` is a valid JSON
          - Add it to the list
  - Group Github workflow runs and Nomad runners in groups indexed by repository url and runner labels
  - For each such group:
    - If there are more Github workflow runs than non-dead Nomad runners:
      - If there are less non-dead Nomad runners than CONFIG.max_runners
        - Generate Nomad job specification from Jinja2 template.
        - Start the job
    - If there are less Github workflow runs than non-dead Nomad runners:
      - Stop a Nomad job
    - For each dead Nomad job:
      - purge it when associated timeout if reached

# How start the scheduler?

Create a file `config.yml` with the following content:

    ---
    github:
      token: the_access_token_or_set_GH_TOKEN_env_variable
    repos:
      # for single repos
      - user/repo1
      - user/repo2
      # for all repositories of this user
      - user

Run the command line:

    nomadtools githubrunner -c ./config.yml run

The default configuration can be listed with `nomadtools githubrunner -c $'{}\n' dumpconfig`.

The configuration description is in source code.

# Nomad job template

The following variables are available in the Jinja2 template:

- CONFIG
  - The whole parsed YAML configuration.
- RUN
  - Dictionary with important generated values.
    - RUN.RUNNER_NAME
      - The Nomad job name and runner name, to be consistent.
    - RUN.REPO_URL
      - The repository the runner should connect to.
    - RUN.LABELS
      - Comma separated list of labels the runner should register to.
    - RUN.REPO
      - Like REPO_URL but without domain
    - RUN.INFO
      - Generated string with some information about the state of scheduler at the time the runner was requested to run.
- RUNSON
  - The `runson:` labels in the workflow are parsed with `shlex.split` `k=v` values
  - For example, `nomadtools mem=1000` results in RUNSON={"nomadtools": "", "mem": "1000"}.
  - The values used by the default template:
    - RUNSON.tag
      - docker image tag
    - RUNSON.cpu
    - RUNSON.mem
    - RUNSON.maxmem
- SETTINGS
  - Dictionary composed of, in order:
    - CONFIG.template_default_settings
    - CONFIG.template_settings
- nomadlib.escape
  - Replace `${` by `$${` and `%{` by `%%{` . For embedding string in template.

# How to use it from Github workflow?

    jobs:
      build:
        runs_on: nomadtools mem=16000

would template the job with `RUNSON={"nomadtools": "", "mem": "16000"}` .

The value of `RUNSON.mem` is then used as the memory settings in the Jinja2 template.

## Multiple elements in runs_on

I do not know what will happen with multiple elements in runs_on, like:

    runs_no:
       - self-hosted
       - nomadtools mem=16000

## Comma in labels

The config.sh github runner configuration script takes comma separated list of elements.

So comma will split the labels into multiple.

Do not use comma in `runs_on:`.

# Example workflow:

https://github.com/Kamilcuk/runnertest/tree/main/.github/workflows
