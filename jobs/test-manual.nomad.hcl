variable "count1" {
  type    = number
  default = 1
}
variable "count2" {
  type = number
  default = 1
}
variable "exit1" {
  type = number
  default = 0
}
variable "exit2" {
  type = number
  default = 0
}

job "test-manual" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  group "group1" {
    count = var.count1
    reschedule { attempts = 0 }
    restart { attempts = 0 }
    task "task1" {
      driver = "docker"
      config {
        image = "busybox:stable"
        args = [
          "sh",
          "-xc",
          "for i in $(seq 3); do echo \"$(date +%Y:%m:%dT%H-%M-%S%z): $i\"; sleep 0.5; done; exit ${var.exit1}"
        ]
        init = true
      }
    }
  }

  group "group2" {
    count = var.count2
    reschedule { attempts = 0 }
    restart { attempts = 0 }
    task "task2" {
      driver = "docker"
      config {
        image = "busybox:stable"
        args = [
          "sh",
          "-xc",
          "for i in $(seq 3); do echo \"$(date +%Y:%m:%dT%H-%M-%S%z): $i\"; sleep 0.5; done; exit ${var.exit2}"
        ]
      }
    }
  }

}
