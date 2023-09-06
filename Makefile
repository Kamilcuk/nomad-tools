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

docker_test:
	docker build .
	timeout 60 docker run --rm "$$(docker build -q .)" bash ./tests/run_in_docker.sh
