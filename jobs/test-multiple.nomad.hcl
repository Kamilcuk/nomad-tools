locals {
  image   = "busybox:stable"
  command = "sh"
  args = [
    "-xeuc",
    <<EOF
    echo 876f767f-7dbb-4e1f-8625-4dcd39f1adaa alloc${NOMAD_ALLOC_INDEX} ${NOMAD_GROUP_NAME} ${NOMAD_TASK_NAME} START
    sleep 0.1
    echo 876f767f-7dbb-4e1f-8625-4dcd39f1adaa alloc${NOMAD_ALLOC_INDEX} ${NOMAD_GROUP_NAME} ${NOMAD_TASK_NAME} STOP
    EOF
  ]
  init  = true
  count = 3
}
job "test-multiple" {
  type = "batch"
  meta {
    uuid = uuidv4()
  }
  reschedule { attempts = 0 }
  group "group1" {
    count = local.count
    restart { attempts = 0 }
    task "task1" {
      driver = "raw_exec"
      config {
        command = local.command
        args    = local.args
      }
    }
    task "task2" {
      driver = "raw_exec"
      config {
        command = local.command
        args    = local.args
      }
    }
  }
  group "group2" {
    count = local.count
    restart { attempts = 0 }
    task "task3" {
      driver = "raw_exec"
      config {
        command = local.command
        args    = local.args
      }
    }
    task "task4" {
      driver = "raw_exec"
      config {
        command = local.command
        args    = local.args
      }
    }
  }
}
