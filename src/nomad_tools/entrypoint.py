import click

from . import (
    nomad_cp,
    nomad_dockers,
    nomad_downloadrelease,
    nomad_gitlab_runner,
    nomad_port,
    nomad_vardir,
    nomad_watch,
)


@click.group()
def cli():
    pass


cli.add_command(nomad_cp.cli)
cli.add_command(nomad_dockers.cli)
cli.add_command(nomad_downloadrelease.cli)
cli.add_command(nomad_gitlab_runner.cli)
cli.add_command(nomad_port.cli)
cli.add_command(nomad_vardir.cli)
cli.add_command(nomad_watch.cli)

if __name__ == "__main__":
    cli()
