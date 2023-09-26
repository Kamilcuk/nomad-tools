job "test-echo10sleep05" {
  type = "batch"
  group "example" {
    task "example" {
      driver = "docker"
      config {
        image = "busybox"
        args = [
          "sh",
          "-xc",
          "for i in $(seq 10); do echo \"$(date +%Y:%m:%dT%H-%M-%S%z): $i\"; sleep 0.5; done"
        ]
        init = true
      }
    }
  }
}
