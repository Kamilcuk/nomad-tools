MAKEFLAGS = -rR --warn-unused-variables
ARGS ?=

.PHONY: test
test:
	env NOMAD_NAMESPACE= NOMAD_TOKEN= pytest -v tests/unit

vagrant_test:
	vagrant up
	vagrant ssh -- -t 'cd /app && ./integration_tests.sh $(ARGS)'

vagrant_destroy:
	vagrant destroy -f

lint: pyright pylava

pyright:
	pyright src/ tests/ $(ARGS)

pylava:
	pylava src/ tests/ $(ARGS)

integration_tests:
	./integration_tests.sh $(ARGS)

paralell_integration_tests:
	./integration_tests.sh -n auto $(ARGS)
