FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . ./

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD alembic upgrade head && uvicorn app.main:app --host ${HOST} --port ${PORT}
