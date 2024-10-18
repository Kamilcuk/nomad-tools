job "nomadtools-githubrunner-executor" {
  type = "batch"
  parameterized {
    meta_required = [
      "ACCESS_TOKEN",
      "REPO_URL",
      "JOB_NAME",
      "LABELS",
      "cpu",
      "mem",
      "memmax",
      "image",
    ]
  }
  meta {
    INFO = <<EOF
This is a runner based on {{ image }} image.
{% if param.docker == "dind" %}
It also starts a docker daemon and is running as privileged
{% elif param.docker == "host" %}
It also mounts a docker daemon from the host it is running on
{% endif %}

EOF
  PARAM = <<EOF
{{param | tojson}}
EOF
  }
  group "nomadtools-githubrunner-executor" {
    reschedule {
      attempts  = 0
      unlimited = false
    }
    restart {
      attempts = 0
      mode     = "fail"
    }

    task "nomadtools-githubrunner-executor" {
      driver       = "docker"
      kill_timeout = "5m"
      config {
        image      = "{{ param.image|default('myoung34/github-runner:latest') }}"
        init       = true
        entrypoint = ["bash", "/local/startscript.sh"]

        mount {
          type     = "bind"
          source   = "/home/kamil/myprojects/nomad-tools/deploy/cachedir"
          target   = "/_work"
          readonly = false
        }

        privileged = true
        # mount {
        #     type   = "bind"
        #     source = "/var/run/docker.sock"
        #     target = "/var/run/docker.sock"
        # }
      }
      template {
        destination     = "local/startscript.sh"
        change_mode     = "noop"
        left_delimiter  = "QWEQWEQEWQEEQ"
        right_delimiter = "ASDASDADSADSA"
        data            = <<EOF
replace(file("./startscript.sh"), "$${", "$$${")
EOF
      }
      env {
        ACCESS_TOKEN        = "${NOMAD_META_ACCESS_TOKEN}"
        REPO_URL            = "${NOMAD_META_REPO_URL}"
        RUNNER_NAME         = "${NOMAD_META_JOB_NAME}"
        LABELS              = "${NOMAD_META_LABELS}"
        RUNNER_SCOPE        = "repo"
        DISABLE_AUTO_UPDATE = "true"
        # RUN_AS_ROOT         = "false"
        EPHEMERAL           = "true"
        START_DOCKER_SERVICE = "true"
        # DEBUG = "true"
      }
      resources {
        cpu         = NOMAD_META_cpu
        memory      = NOMAD_META_mem
        memory_max = NOAMD_META_memmax
      }
    }
  }
}
