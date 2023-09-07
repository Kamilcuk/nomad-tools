FROM python:3.7
RUN apt-get update -y && apt-get install -y --no-install-recommends sudo unzip wget
COPY pyproject.toml ./pyproject.toml
RUN pip install click requests tomli pytest
COPY ./tests/install_nomad.sh ./install_nomad.sh
RUN bash ./install_nomad.sh
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY ./src .
COPY ./pyproject.toml .
RUN pip install -e '.[test]'
COPY ./tests/unit ./tests/unit
RUN pytest -sxv ./tests/unit
COPY . .
