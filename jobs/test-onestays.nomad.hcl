locals {
  image   = "busybox:stable"
  command = "sh"
  args = ["-xc", <<EOF
    echo "Hello world from ${NOMAD_JOB_NAME} alloc${NOMAD_ALLOC_INDEX} ${NOMAD_GROUP_NAME} ${NOMAD_TASK_NAME}" > /tmp/index.html
    exec httpd -f -v -p ${NOMAD_PORT_http} -h /tmp
    EOF
  ]
  count = 1
}
job "test-onestays" {
  type = "service"
  reschedule { attempts = 0 }

  group "group1stays" {
    restart { attempts = 0 }
    update {
      canary       = 1
      auto_promote = true
      auto_revert  = true
    }
    network {
      port "http" {}
    }
    task "task1stays" {
      driver = "docker"
      config {
        image   = local.image
        command = local.command
        args    = local.args
        ports   = ["http"]
        init    = true
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

  group "group2change" {
    meta {
      uuid = uuidv4()
    }
    restart { attempts = 0 }
    update {
      canary       = 1
      auto_promote = true
      auto_revert  = true
    }
    network {
      port "http" {}
    }
    task "task2change" {
      driver = "docker"
      config {
        image   = local.image
        command = local.command
        args    = local.args
        ports   = ["http"]
        init    = true
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

  group "group3change" {
    meta {
      uuid = uuidv4()
    }
    restart { attempts = 0 }
    update {
      canary       = 1
      auto_promote = true
      auto_revert  = true
    }
    network {
      port "http" {}
    }
    task "task3change" {
      driver = "docker"
      config {
        image   = local.image
        command = local.command
        args    = local.args
        ports   = ["http"]
        init    = true
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
