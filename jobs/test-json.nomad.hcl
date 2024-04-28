job "test-jsons" {
  group "example" {
    task "server" {
      identity {
        file = true
        ttl  = "1h"
      }
      driver = "raw_exec"
      config {
        command = "echo"
      }
      template {
         source        = "local/redis.conf.tpl"
        destination   = "local/redis.conf"
        change_mode   = "signal"
        change_signal = "SIGINT"
        error_on_missing_key = false
      }
    }
  }
}
