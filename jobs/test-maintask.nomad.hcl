variable "count" {
  default = 2
}
job "test-maintask" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  group "maintask" {
    count = var.count
    task "main" {
      driver = "raw_exec"
      config {
        command = "sh"
        args = ["-xc", "echo main && exec sleep 60"]
      }
    }
    task "prestart" {
      driver = "raw_exec"
      config {
        command = "sh"
        args = ["-xc", "echo prestart"]
      }
      lifecycle {
        hook = "prestart"
      }
    }
    task "prestart_sidecar" {
      driver = "raw_exec"
      config {
        command = "sh"
        args = ["-xc", "echo prestart_sidecar && exec sleep 60" ]
      }
      lifecycle {
        hook = "prestart"
        sidecar = true
      }
    }
    task "poststart" {
      driver = "raw_exec"
      config {
        command = "sh"
        args = ["-xc", "echo poststart && exec sleep 60"]
      }
      lifecycle {
        hook = "poststart"
      }
    }
    task "poststop" {
      driver = "raw_exec"
      config {
        command = "sh"
        args = ["-xc", "echo poststop && exec sleep 1"]
      }
      lifecycle {
        hook = "poststop"
      }
    }
  }
}
