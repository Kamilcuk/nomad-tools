ARG VERSION=alpine
FROM python:${VERSION} AS req
WORKDIR /app
RUN apk add --no-cache git bash
COPY requirements.txt .
RUN pip install --no-cache-dir -r ./requirements.txt

FROM req AS test
COPY ./requirements-test.txt .
RUN pip install --no-cache-dir -r ./requirements-test.txt
COPY . .
RUN pip install --no-cache-dir -e . && nomadtools --version
# RUN ./unit_tests.sh

FROM req AS app
COPY . .
RUN pip install --no-cache-dir -e . && nomadtools --version
ENTRYPOINT ["nomadtools"]
