variable "ok" {
  type    = bool
  default = true
}
job "test-upgrade1" {
  type = "service"

  group "group1" {
    count = 1
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
    task "task1" {
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
            env > /tmp/index.html &&
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

  group "group2" {
    count = 1
    meta {
      uuid = uuidv4()
    }
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
    task "task2" {
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
            env > /tmp/index.html &&
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

  group "group3" {
    count = 1
    meta {
      uuid = uuidv4()
    }
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
    task "task3" {
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
            env > /tmp/index.html &&
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
