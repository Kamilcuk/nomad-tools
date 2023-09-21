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

run = USERGROUP=$(shell id -u):$(shell id -g) docker compose run --build --rm
docker_test:
	$(run) test $(ARGS)

docker_shell:
	@mkdir -vp build
	 $(run) -i shell $(ARGS)

docker_test_parallel: ARGS = -nauto
docker_test_parallel: docker_test
