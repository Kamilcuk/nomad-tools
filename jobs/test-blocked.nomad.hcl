variable "block" {
  type    = bool
  default = false
}
job "test-blocked" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  group "blocked" {
    reschedule {
      attempts  = 0
      unlimited = false
    }
    task "blocked" {
      driver = "raw_exec"
      restart {
        attempts = 0
      }
      config {
        command = "sh"
        args = [ "-xc", "true" ]
      }
      resources {
        memory = var.block ? 9999999 : 300
      }
    }
  }
}
