FROM node:22-slim

RUN npm install -g @earendil-works/pi-coding-agent

RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY . .
RUN uv sync --frozen

RUN mkdir -p /app/pi-config /app/hf-cache

# Makes main.py run on startup - not ideal for using headless (detached) mode
#CMD ["uv", "run", "main.py"]
# So instead set it to sleep forever, staying alive without being used. (keeps container running, not waiting)
CMD ["sleep", "infinity"]