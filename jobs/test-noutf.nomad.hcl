job "test-noutf" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  group "noutf" {
    reschedule { attempts = 0 }
    restart { attempts = 0 }
    task "noutf" {
      driver = "raw_exec"
      config {
        command = "sh"
        args = ["-xc", <<EOF
          printf "stdout 0xc0 byte: \xc0\n"
          printf "stderr 0xc0 byte: \xc0\n" >&2
          EOF
        ]
      }
    }
  }
}
