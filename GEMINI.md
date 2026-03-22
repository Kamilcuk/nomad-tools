# Gemini Code Companion Context

## Project Overview

This project, `nomad-tools`, is a Python-based command-line utility designed to simplify interaction with HashiCorp Nomad. It provides a collection of tools for managing and monitoring Nomad jobs, nodes, and variables. The project is intended to be used as a CLI tool and can be installed via pip.

The main technologies used are Python, with dependencies managed by `pip` and defined in `requirements.txt` and `requirements-test.txt`. The project is structured as a standard Python package with source code in `src/nomad_tools`.

## Building and Running

### Installation and Setup

To set up the development environment, install the package in editable mode with test dependencies:

```bash
pip install -e '.[test]'
```

### Running Tests

The project has both unit and integration tests.

-   **Unit Tests:** Run the unit tests with the following command:

    ```bash
    ./unit_tests.sh
    ```

-   **Integration Tests:** Integration tests require a running Nomad server. They can be executed with:

    ```bash
    ./integration_tests.sh
    ```

    You can also run specific integration tests by passing arguments:

    ```bash
    ./integration_tests.sh -k nomad_vardir
    ```

## Development Conventions

### Code Style and Linting

The project uses several tools to maintain code quality:

-   `pyright` for type checking.
-   `pylava` for linting.
-   `ruff` for linting and auto-formatting.

You can run the linters with the following command:

```bash
make lint
```

To automatically fix some of the linting issues, you can use:

```bash
make autofix
```

### Contribution Guidelines

Contributions are welcome. The `README.md` encourages creating GitHub issues for feature requests and bugs, and pull requests for contributions.
