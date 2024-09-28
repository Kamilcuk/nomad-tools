MAKEFLAGS = -rR --warn-unused-variables
ARGS ?=
VERBOSE ?=

.PHONY: test
test:
	env NOMAD_NAMESPACE= NOMAD_TOKEN= pytest -v tests/unit

vagrant_test:
	vagrant up
	vagrant ssh -- -t 'cd /app && ./integration_tests.sh $(ARGS)'

vagrant_destroy:
	vagrant destroy -f

lint: pyright pylava ruff

pyright:
	pyright src/ tests/ $(ARGS)

pylava:
	pylava src/ tests/ $(ARGS)

ruff:
	ruff check src/ tests/ $(ARGS)

integration_tests:
	./integration_tests.sh $(ARGS)

paralell_integration_tests:
	./integration_tests.sh -n auto $(ARGS)

importtime: ARGS=entry
importtime:
	python -X importtime -c 'import nomad_tools.$(ARGS)' 2>build/importtime.log
	tuna build/importtime.log

try_docker_integration_test:
	docker build --target integration -f ./tests/Dockerfile.integration_test -t tmp .
	docker run --network=host --privileged -ti --rm tmp \
		bash -xeuc ' \
			dockerd & \
			./tests/provision.sh nomad_start & \
			sleep 10 && \
			./integration_tests.sh \
		'

run_githubrunner_locally:
	. ./.env && nomaddtools githubrunner -c ./deploy/githubrunner.yml $(ARGS) run

run_githubrunner:
	. ./.env && nomadtools watch run \
			-var GITHUB_TOKEN=$$GITHUB_TOKEN \
			-var NOMAD_TOKEN=$$NOMAD_TOKEN \
			-var NOMAD_NAMESPACE=$$NOMAD_NAMESPACE \
			-var VERBOSE=$(VERBOSE) \
			./deploy/nomadtools-githubrunner.nomad.hcl
