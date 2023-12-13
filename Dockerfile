ARG VERSION=latest
FROM python:${VERSION} AS req
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt

FROM req AS app
COPY . .
RUN pip install --no-cache-dir -e . && nomad-watch --version

FROM req AS test
COPY ./requirements-dev.txt .
RUN pip install --no-cache-dir -r ./requirements-test.txt
COPY . .
RUN pip install --no-cache-dir -e . && nomad-watch --version
RUN ./unit_tests.sh
