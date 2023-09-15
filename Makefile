MAKEFLAGS = -rR --warn-unused-variables
ARGS ?=

.PHONY: all
all:
	echo

.PHONY: deps
deps:
	if ! python -c 'import click, requests'; then sudo pacman -S --noconfirm --needed python-click python-requests; fi

.PHONY: e
e:
	pip install --break-system-packages -e .

.PHONY:testdeps
testdeps:
	pip install --break-system-packages -e '.[test]'

.PHONY: test
test:
	env NOMAD_NAMESPACE= NOMAD_TOKEN= pytest -v tests/unit

test_integration:
	pytest -sxv tests/integration -p no:cacheprovider $(ARGS)

tty = $(shell test -t 0 && printf -- -t)
docker_run = docker run $(tty) --rm --name nomad_tools_test
docker_id = "$$(docker build -q .)"
docker_test:
	docker build .
	time timeout -v 30 $(docker_run) $(docker_id)  \
		timeout 30 bash ./tests/run_in_docker.sh $(ARGS)

docker_shell:
	docker build --target requirements .
	$(docker_run) -i -u $(shell id -u):$(shell id -g) \
		-v $(PWD):/app -w /app \
		-e HOME=/tmp \
		$$(docker build --target requirements -q .) \
		bash -lc \
		'export PATH=/tmp/.local/bin:$$PATH; /app/tests/start_nomad_server.sh & pip install -e . ; wait; bash'

docker_test_parallel: ARGS = -nauto
docker_test_parallel: docker_test
