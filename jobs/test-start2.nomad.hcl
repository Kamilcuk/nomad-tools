variable "script" {
  type = string
  default = "echo hello world"
}
job "test-start2" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  group "start2" {
    reschedule {
      attempts = 0
    }
    restart {
      attempts = 0
    }
    task "start2" {
      driver = "raw_exec"
      config {
        command = "sh"
        args = ["-xc", var.script]
      }
    }
  }
}
