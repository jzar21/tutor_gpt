FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p ./data

EXPOSE 8080

CMD ["python", "app.py", "--config_path", "custom_configs/config_llama.json"]
CMD ["open-webui", "serve"]
