job "test-forever" {
  # meta { uuid = uuidv4() }
  group "test-forever" {
    task "test-forever" {
      driver = "docker"
      config {
        image   = "busybox:stable"
        command = "sleep"
        args    = ["3600h"]
      }
    }
  }
}
