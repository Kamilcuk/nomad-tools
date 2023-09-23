job "example" {
  datacenters = ["*"]
  type = "service"
  group "cache" {
    task "redis" {
      driver = "docker"
      config {
        image = "busybox"
        args=["sleep", "infinity"]
      }
    }
  }
}
