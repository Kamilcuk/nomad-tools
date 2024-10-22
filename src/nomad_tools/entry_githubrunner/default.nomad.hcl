locals {
    INFO = <<EOFEOF
This is a default runner shipped with nomadtools based on myoung34/github-runner image.

User requested labels were parsed to the following:
  {{ nomadlib.escape(RUNSON | tojson) }}

The following parameters were generated for this job:
  {{ nomadlib.escape(RUN | tojson) }}

Job is running with the following settings:
  {{ nomadlib.escape(SETTINGS | tojson) }}

{% if SETTINGS.docker == "dind" %}
The container runs with --privileged and starts a docker-in-docker instance.
{% elif SETTINGS.docker == "host" %}
The container mounts the /var/run/docker.sock from the host.
{% endif %}
{% if RUN.nodocker is defined %}
The user requested to run without docker.
{% endif %}

EOFEOF
}

job "{{ RUN.RUNNER_NAME }}" {
  {{ SETTINGS.extra_job }}
  type = "batch"

  group "{{ RUN.RUNNER_NAME }}" {
    {{ SETTINGS.extra_group }}

    reschedule {
      attempts  = 0
      unlimited = false
    }
    restart {
      attempts = 0
      mode     = "fail"
    }

    task "{{ RUN.RUNNER_NAME }}" {
      {{ SETTINGS.extra_task }}

      driver       = "docker"
      kill_timeout = "5m"
      config {
        {{ SETTINGS.extra_config }}

        image      = "myoung34/github-runner:{{ RUNSON.tag | default('latest') }}"
        init       = true
{% if SETTINGS.entrypoint %}
        entrypoint = ["${NOMAD_TASK_DIR}/nomadtools_entrypoint.sh"]
{% endif %}

{% if SETTINGS.cachedir %}
        mount {
          type     = "bind"
          source   = "{{ SETTINGS.cachedir }}"
          target   = "/_work"
          readonly = false
        }
{% endif %}

{% if not RUNSON.nodocker %}
  {% if SETTINGS.docker == "dind" %}
        privileged = true
  {% elif SETTINGS.docker == "host" %}
        mount {
            type   = "bind"
            source = "/var/run/docker.sock"
            target = "/var/run/docker.sock"
        }
  {% endif %}
{% endif %}

      }

      env {
        ACCESS_TOKEN         = "{{ SETTINGS.access_token or CONFIG.github.token }}"
        REPO_URL             = "{{ RUN.REPO_URL }}"
        RUNNER_NAME          = "{{ RUN.RUNNER_NAME }}"
        RUNSON               = "{{ RUN.LABELS }}"
        RUNNER_SCOPE         = "repo"
        DISABLE_AUTO_UPDATE  = "true"
{% if not SETTINGS.run_as_root %}
        RUN_AS_ROOT          = "false"
{% endif %}
{% if SETTINGS.ephemeral %}
        EPHEMERAL            = "true"
{% endif %}
{% if not RUNSON.nodocker %}
  {% if SETTINGS.docker == "dind" %}
        START_DOCKER_SERVICE = "true"
  {% endif %}
{% endif %}
{% if SETTINGS.debug %}
        DEBUG_OUTPUT         = "true"
{% endif %}
      }

      resources {
{% if RUNSON.cpu %}
        cpu        = {{ RUNSON.cpu }}
{% endif %}
{% if RUNSON.mem %}
        memory     = {{ RUNSON.mem }}
{% endif %}
{% if RUNSON.maxmem %}
        memory_max = {{ RUNSON.maxmem }}
{% endif %}
      }

{% if SETTINGS.entrypoint %}
      template {
        destination     = "local/nomadtools_entrypoint.sh"
        change_mode     = "noop"
        left_delimiter  = "QWEQWEQEWQEEQ"
        right_delimiter = "ASDASDADSADSA"
        perms           = "755"
        data            = <<EOFEOF
{{ nomadlib.escape(SETTINGS.entrypoint) }}
EOFEOF
      }
{% endif %}

    }
  }
}
