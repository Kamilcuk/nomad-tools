MAKEFLAGS = -rR --warn-unused-variables

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

ARGS ?=
docker_test:
	docker build .
	time timeout -v 30 docker run $(shell test -t 0 && printf -- -t) --rm "$$(docker build -q .)" \
		timeout 30 bash ./tests/run_in_docker.sh $(ARGS)

docker_test_parallel: ARGS = -nauto
docker_test_parallel: docker_test
