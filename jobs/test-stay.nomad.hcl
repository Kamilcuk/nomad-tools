variable "script" {
  type    = string
  default = "sleep infinity"
}
job "test-stay" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }

  group "docker" {
    reschedule { attempts = 0 }
    restart { attempts = 0 }
    task "docker" {
      driver = "docker"
      config {
        image   = "busybox:stable"
        command = "sh"
        args    = ["-xc", var.script]
      }
    }
  }

  group "raw_exec" {
    reschedule { attempts = 0 }
    restart { attempts = 0 }
    task "raw_exec" {
      driver = "raw_exec"
      config {
        command = "sh"
        args    = ["-xc", var.script]
      }
    }
  }

}
