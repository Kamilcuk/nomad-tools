job "{{ JOB_NAME }}" {
  namespace = "{{ config.NOMAD_NAMESPACE or 'gitlabrunner' }}"
  type = "batch"
  meta {
    # These variables are for easy navigation documenation in the Nomad gui.
    CI_JOB_URL: "{{ CUSTOM_ENV_CI_JOB_URL }}"
    CI_JOB_NAME: "{{ CUSTOM_ENV_CI_JOB_NAME }}"
    CI_PROJECT_URL: "{{ CUSTOM_ENV_CI_PROJECT_URL }}"
    CI_DRIVER: "{{ task.Driver }}"
    CI_RUNUSER: "{{ dc.user }}"
    CI_OOM_SCORE_ADJUST: "{{ config.oom_score_adjust }}"
    CI_CPUSET_CPUS: "{{ cpuset_cpus }}"
  }
  group "R" {
    reschedule {
      attempts  = 0
      unlimited = false
    }
    restart {
      attempts = 0
      mode     = "fail"
    }
    task "ci-task" {
      env {
        {% for k, v in CUSTOM_ENV %}
        {{ k }}: {{ quote(v) }}
        {% endfor %}
      }
      resources {
        CPU = "{{ [CUSTOM_ENV_NOMADRUNNER_CPU|int, config.CPU]|min }}"
                "CPU": min_none(int(cenv.get("NOMADRUNNER_CPU", 0)), self.CPU),
                "Cores": min_none(int(cenv.get("NOMADRUNNER_CORES", 0)), self.cores),
                "MemoryMB": min_none(
                    int(cenv.get("NOMADRUNNER_MEMORY_MB", 0)), self.MemoryMB
                ),
                "MemoryMaxMB": min_none(
                    int(cenv.get("NOMADRUNNER_MEMORY_MAX_MB", 0)),
                    self.MemoryMaxMB,
                ),
    }
  }
}
