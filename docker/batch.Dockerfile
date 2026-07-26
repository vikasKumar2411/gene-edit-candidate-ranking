FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/program/src

WORKDIR /opt/program

COPY docker/batch-requirements.txt /opt/program/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /opt/program/requirements.txt

COPY src /opt/program/src
COPY config /opt/program/config

ENTRYPOINT ["python", "-m", "gene_edit_ranking.inference.batch_entrypoint"]
