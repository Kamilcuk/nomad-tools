ARG PYTHON=3.12
FROM python:${PYTHON}-slim AS req
WORKDIR /app
RUN set -x && \
  apt-get update && \
  apt-get install -y --no-install-recommends git bash && \
  rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt

FROM req AS test
COPY ./requirements-test.txt .
RUN pip install --no-cache-dir -r ./requirements-test.txt
COPY ./pyproject.toml ./pyproject.toml
COPY ./src ./src
COPY ./.git ./.git
RUN pip install --no-cache-dir -e . && nomadtools --version
COPY . .
RUN ./unit_tests.sh

FROM req AS pyright
RUN pip install pyright && pyright --version
COPY ./pyproject.toml ./pyproject.toml
COPY src/ src/
RUN pyright src/
COPY ./requirements-test.txt .
RUN pip install --no-cache-dir -r ./requirements-test.txt
COPY tests/ tests/
RUN pyright tests/

FROM req AS app
COPY ./pyproject.toml ./pyproject.toml
COPY ./src ./src
COPY ./.git ./.git
RUN pip install --no-cache-dir -e . && nomadtools --version
RUN nomadtools downloadrelease -p 1.8.2 nomad /bin/nomad && nomad --version
COPY . .
ENTRYPOINT ["nomadtools"]
