variable "ok" {
  type    = bool
  default = true
}
job "test-listen" {
  type = "service"
  meta {
    uuid = uuidv4()
  }
  group "listen" {
    network {
      port "http" {
      }
    }
    reschedule {
      attempts  = 0
      interval  = "15s"
      delay     = "5s"
      unlimited = false
    }
    update {
      canary       = 1
      auto_revert  = true
      auto_promote = true
    }
    task "listen" {
      driver = "docker"
      restart {
        attempts = 0
        interval = "15s"
        delay    = "1s"
      }
      config {
        image   = "busybox:stable"
        command = "sh"
        args = [
          "-xc",
          var.ok ? <<EOF
            echo "test-listen $(hostname)" > /tmp/index.html &&
            exec httpd -f -v -p ${NOMAD_PORT_http} -h /tmp
          EOF
          : "exec false",
        ]
        ports = ["http"]
        init  = true
      }
      service {
        port     = "http"
        provider = "nomad"
        check {
          type     = "http"
          path     = "/"
          interval = "1s"
          timeout  = "1s"
        }
      }
    }
  }
}
