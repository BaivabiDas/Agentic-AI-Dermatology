FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir psutil

CMD ["python", "selection_agent.py"]
