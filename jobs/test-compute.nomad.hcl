variable "input" {
  type = string
  default = ""
}
variable "exit" {
  type = number
  default = 4
}
variable "wait" {
  type = number
  default = 2
}
job "test-compute" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  group "compute" {
    reschedule { attempts = 0 }
    restart { attempts = 0 }
    task "compute" {
      driver = "docker"
      config {
        image = "busybox:stable"
        args = [
          "sh",
          "-c",
          <<EOF
echo "INFO:__main__: Doing hard calculations...."
sleep ${var.wait}
echo "INFO:__main__: Computing the meaning of life...."
sleep ${var.wait}
echo "INFO:__main__: Adding delay for more drama..."
sleep ${var.wait}
echo "ERROR:__main__: failure, exiting with 4 exit status!"
exit ${var.exit}
          EOF
        ]
        init = true
      }
    }
  }
}
