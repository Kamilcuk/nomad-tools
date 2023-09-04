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
	pytest -v
