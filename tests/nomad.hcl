# Example nomad server configuration used for testing this project.
bind_addr            = "0.0.0.0"
disable_update_check = true
server {
  enabled          = true
  bootstrap_expect = 1
}
client {
  enabled                  = true
  gc_disk_usage_threshold  = 100
  gc_inode_usage_threshold = 100
  drain_on_shutdown {
    deadline           = "1m"
    force              = true
    ignore_system_jobs = false
  }
}
plugin "raw_exec" {
  config {
    enabled = true
  }
}
plugin "docker" {
  config {
    allow_privileged = true
    gc {
      image = false
    }
    volumes {
      enabled = true
    }
  }
}
consul {
  client_auto_join = false
  server_auto_join = false
}
