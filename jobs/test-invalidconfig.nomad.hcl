job "test-invalidconfig" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  reschedule { attempts = 0 }
  group "group1" {
    restart { attempts = 0 }
    task "task1" {
      driver = "docker"
      config {
        invalid = "something_invalid"
      }
    }
  }
}
