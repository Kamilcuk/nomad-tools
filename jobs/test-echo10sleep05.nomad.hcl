job "test-echo10sleep05" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  group "example" {
    task "example" {
      driver = "docker"
      config {
        image = "busybox:stable"
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
