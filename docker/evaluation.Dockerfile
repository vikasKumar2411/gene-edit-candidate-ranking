FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/program/src

WORKDIR /opt/program

COPY docker/training-requirements.txt /opt/program/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /opt/program/requirements.txt

COPY src /opt/program/src
COPY config /opt/program/config
COPY pyproject.toml /opt/program/pyproject.toml

ENTRYPOINT ["python", "-m", "gene_edit_ranking.training.evaluate_model"]
