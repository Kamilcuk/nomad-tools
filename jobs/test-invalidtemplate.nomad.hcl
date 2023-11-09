job "test-invalidconfig" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  reschedule { attempts = 0 }
  group "group1" {
    restart { attempts = 0 }
    task "task1" {
      driver = "raw_exec"
      config {
        command = "sh"
      }
      template {
        destination = "local/data.txt"
        data        = "{{ invalid"
      }
    }
  }
}
