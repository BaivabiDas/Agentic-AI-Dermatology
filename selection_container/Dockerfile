FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --no-deps -r requirements.txt


CMD ["python", "selection_agent.py"]
