locals {
  DIR = "${abspath(".")}"
}
variable "GITHUB_TOKEN" {
  type = string
}
variable "NOMAD_TOKEN" {
  type = string
}
variable "NOMAD_NAMESPACE" {
  type = string
}
variable "VERBOSE" {
  type = string
}

job "nomadtools-githubrunner" {
  namespace = "github"
  group "nomadtools-githubrunner" {
    ephemeral_disk {
      migrate = true
      sticky = true
    }
    restart {
      mode = "delay"
    }

    task "build" {
      driver = "docker"
      config {
        image = "docker:cli"
        args = ["sh", "-xc", "docker build -t nomad:${NOMAD_ALLOC_ID} --target app ."]
        work_dir = "/mnt"
        mount {
          type = "bind"
          source = "/var/run/docker.sock"
          target = "/var/run/docker.sock"
        }
        mount {
          type = "bind"
          source = local.DIR
          target = "/mnt"
        }
      }
      lifecycle {
        hook = "prestart"
      }
    }

    task "nomadtools-githubrunner" {
      driver = "docker"
      config {
        image = "nomad:${NOMAD_ALLOC_ID}"
        network_mode = "host"
        entrypoint = []
        args = [ "sh", "-xc", <<EOF
          nomadtools githubrunner \
            $([ -n ${var.VERBOSE} ] && echo --verbose) \
            --config=${NOMAD_TASK_DIR}/githubrunner.yml \
            run
          EOF
        ]
        mount {
          type     = "bind"
          source   = "${local.DIR}/deploy"
          target   = "/root/.cache/nomadtools"
          readonly = false
        }
      }
      env {
        GITHUB_TOKEN = var.GITHUB_TOKEN
        NOMAD_TOKEN = var.NOMAD_TOKEN
        NOMAD_NAMESPACE = var.NOMAD_NAMESPACE
      }
      template {
        destination = "local/githubrunner.yml"
        data        = file("./deploy/githubrunner.yml")
        change_mode = "noop"
      }
    }

  }
}

