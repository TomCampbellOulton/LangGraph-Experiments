FROM nvidia/cuda:13.2.1-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

EXPOSE 8080

ENTRYPOINT ["/src/build/bin/llama-server"]