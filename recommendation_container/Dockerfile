
FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir flask ollama

EXPOSE 5002

CMD ["python", "recommendation_agent.py"]
