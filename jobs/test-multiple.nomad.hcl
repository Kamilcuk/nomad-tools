locals {
  image   = "busybox:stable"
  command = "sh"
  args = [
    "-xeuc",
    <<EOF
    echo 876f767f-7dbb-4e1f-8625-4dcd39f1adaa ${NOMAD_GROUP_NAME} ${NOMAD_TASK_NAME} START
    sleep 0.1;
    echo 876f767f-7dbb-4e1f-8625-4dcd39f1adaa ${NOMAD_GROUP_NAME} ${NOMAD_TASK_NAME} STOP
    EOF
  ]
  init = true
}
job "test-multiple" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  reschedule { attempts = 0 }
  group "group1" {
    restart { attempts = 0 }
    task "task1" {
      driver = "docker"
      config {
        image   = local.image
        command = local.command
        args    = local.args
        init    = local.init
      }
    }
    task "task2" {
      driver = "docker"
      config {
        image   = local.image
        command = local.command
        args    = local.args
        init    = local.init
      }
    }
  }
  group "group2" {
    restart { attempts = 0 }
    task "task3" {
      driver = "docker"
      config {
        image   = local.image
        command = local.command
        args    = local.args
        init    = local.init
      }
    }
    task "task4" {
      driver = "docker"
      config {
        image   = local.image
        command = local.command
        args    = local.args
        init    = local.init
      }
    }
  }
}
