job "test-upgrade2" {
  group "upgrade2" {
    task "upgrade2" {
      meta {
        index = "-1"
      }
      driver = "raw_exec"
      config {
        command = "bash"
        args = ["-xc", <<EOF
          echo "start $(date)"
          sleep infinity
          trap 'echo "stop  $(date)"' EXIT
          EOF
        ]
      }
    }
  }

  group "disabled" {
    count = 0
    meta {
      index = uuidv4()
    }
    task "upgrade2" { driver = "invalid" }
  }
}
