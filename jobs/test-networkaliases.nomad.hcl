locals {
  env = {
    DOCKER_TLS_CERTDIR = "/certs"
    DOCKER_CERT_PATH   = "/certs/client"
    DOCKER_HOST        = "tcp://docker:2376"
    DOCKER_TLS_VERIFY  = "1"
  }
}
job "test-networkaliases" {
  group "a" {
    network {
      mode = "bridge"
    }
    task "a" {
      driver = "docker"
      config {
        image           = "docker:dind"
        network_aliases = ["docker"]
        privileged      = true
        volumes         = ["${NOMAD_ALLOC_DIR}:/certs"]
      }
      env = local.env
    }
    task "b" {
      driver = "docker"
      config {
        image   = "docker:cli"
        command = "sh"
        args = ["-xc", <<EOF
          while ! docker info >/dev/null; do sleep 1; done
          docker info
          docker run --rm alpine echo hello world
          sleep 1000
          EOF
        ]
        volumes = ["${NOMAD_ALLOC_DIR}:/certs"]
      }
      env = local.env
    }
  }
}
