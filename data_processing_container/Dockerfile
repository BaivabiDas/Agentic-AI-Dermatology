FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    flask \
    numpy \
    pillow \
    torch \
    torchvision

EXPOSE 5000

CMD ["python", "data_processing_agent.py"]