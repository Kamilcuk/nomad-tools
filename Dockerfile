FROM python:3.7 AS requirements
RUN set -x && \
	apt-get update -y && \
	apt-get install -y --no-install-recommends sudo unzip wget && \
	rm -rf /var/lib/apt/lists/*
RUN set -x && \
	pip install --no-cache-dir --upgrade pip && \
	pip install --no-cache-dir click requests tomli pytest pytest-xdist
WORKDIR /app
COPY ./tests/install_nomad.sh ./tests/install_nomad.sh
RUN bash ./tests/install_nomad.sh
ENV PYTHONUNBUFFERED=1

FROM requirements AS app
COPY ./src .
COPY ./pyproject.toml .
RUN pip install --no-cache-dir -e '.[test]'
COPY ./tests/unit ./tests/unit
RUN pytest -sxv ./tests/unit
COPY . .
