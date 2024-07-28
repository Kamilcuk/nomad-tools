locals {
  DIR = "${abspath(".")}"
}
variable "GITHUB_TOKEN" {
  type = string
}
variable "NOMAD_TOKEN" {
  type = string
}
job "nomadtools-githubrunner" {
  namespace = "github"
  group "nomadtools-githubrunner" {
    ephemeral_disk {
      migrate = true
      sticky = true
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
        args = [
          "githubrunner",
          "--config",
          <<-EOF
---
nomad:
  namespace: github
  token: ${var.NOMAD_TOKEN}
github:
  token: ${var.GITHUB_TOKEN}
  cachefile:  ${NOMAD_ALLOC_DIR}/data/githubcache.json
opts:
  docker: host
repos:
  - Kamilcuk
runner_inactivity_timeout: 1w
EOF
          ,
          "run",
        ]
      }
    }

  }
}

