FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH="/app/src:$PYTHONPATH"

ENV PORT=3002
EXPOSE 3002

WORKDIR src/spel

# Download all necessary files, prime the model
RUN python prime.py || true

CMD ["python", "server.py", "spel", "n"]

# docker build --platform=linux/amd64 -t spel .
# docker run -t -p 8002:3002 spel