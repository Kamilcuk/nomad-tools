job "test-run0" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  group "start" {
    reschedule { attempts = 0 }
    restart { attempts = 0 }
    task "start" {
      driver = "raw_exec"
      config {
        command = "sh"
        args = ["-xc", "echo hello world"]
      }
    }
  }
}
