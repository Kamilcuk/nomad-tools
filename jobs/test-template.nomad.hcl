variable "script" {
  type = string
  default = ""
}
job "test-${NAME}" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  group "${NAME}" {
    reschedule {
      attempts = 0
    }
    restart {
      attempts = 0
    }
    task "${NAME}" {
      driver = "raw_exec"
      config {
        command = "sh"
        args = ["-xc", <<-EOF
          ${SCRIPT}
          ${var.script}
        EOF
        ]
      }
    }
  }
}
