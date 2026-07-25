FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SM_MODEL_DIR=/opt/ml/model
ENV PYTHONPATH=/opt/program/src

WORKDIR /opt/program

COPY docker/inference-requirements.txt /opt/program/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /opt/program/requirements.txt

COPY src /opt/program/src

EXPOSE 8080

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "60", "gene_edit_ranking.inference.serve:app"]
