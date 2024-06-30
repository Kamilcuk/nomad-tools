job "test-sigterm" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  group "example" {
    task "example" {
      driver = "docker"
      config {
        image = "bash"
        args = [
          "sh",
          "-xc",
          <<EOF
          trap 'echo SIGTERM; wait' SIGTERM
          trap 'echo EXIT' EXIT
          sleep infinity &
          wait
          EOF
        ]
      }
      kill_timeout = "20s"
      shutdown_delay = "10s"
    }
  }
}
