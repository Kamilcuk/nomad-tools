job "{{ param.JOB_NAME }}" {
  type = "batch"
  meta {
    INFO = <<EOF
This is a runner based on {{ image }} image.
{% if param.docker == "dind" %}
It also starts a docker daemon and is running as privileged
{% elif param.docker == "host" %}
It also mounts a docker daemon from the host it is running on
{% endif %}

EOF

{% if param.debug %}
  PARAM = <<EOF
{{param | tojson}}
EOF
{% endif %}
  }
  group "{{ param.JOB_NAME }}" {
    reschedule {
      attempts  = 0
      unlimited = false
    }
    restart {
      attempts = 0
      mode     = "fail"
    }

    task "{{ param.JOB_NAME }}" {
      driver       = "docker"
      kill_timeout = "5m"
      config {
        image      = "{{ param.image | default('myoung34/github-runner:latest') }}"
        init       = true
        entrypoint = ["bash", "/local/startscript.sh"]

        {% if param.cachedir %}
        mount {
          type     = "bind"
          source   = "{{ param.cachedir }}"
          target   = "/_work"
          readonly = false
        }
        {% endif %}

        {% if param.docker == "dind" %}
        privileged = true
        {% elif param.docker == "host" %}
        mount {
            type   = "bind"
            source = "/var/run/docker.sock"
            target = "/var/run/docker.sock"
        }
        {% endif %}

        {{ param.extra_config }}
      }
      template {
        destination     = "local/startscript.sh"
        change_mode     = "noop"
        left_delimiter  = "QWEQWEQEWQEEQ"
        right_delimiter = "ASDASDADSADSA"
        data            = <<EOF
{% if not param.startscript %}{{ 1/0 }}{% endif %}
{{ escape(param.startscript) }}
EOF
      }
      env {
        ACCESS_TOKEN        = "{{ CONFIG.github.token }}"
        REPO_URL            = "{{ param.REPO_URL }}"
        RUNNER_NAME         = "{{ param.JOB_NAME }}"
        LABELS              = "{{ param.LABELS }}"
        RUNNER_SCOPE        = "repo"
        DISABLE_AUTO_UPDATE = "true"
        # RUN_AS_ROOT         = "false"

        {% if param.ephemeral == "true" %}
        EPHEMERAL           = "true"
        {% endif %}
        {% if param.docker == "dind" %}
        START_DOCKER_SERVICE = "true"
        {% endif %}
        {% if param.debug %}
        DEBUG = "true"
        {% endif %}
      }
      resources {
        {% if param.cpu %}
        cpu         = {{ param.cpu }}
        {% endif %}
        {% if param.mem %}
        memory      = {{ param.mem }}
        {% endif %}
        {% if param.maxmem %}
        memory_max = {{ param.maxmem }}
        {% endif %}
      }
      {{ param.extra_task }}
    }
    {{ param.extra_group }}
  }
  {{ param.extra_job }}
}
