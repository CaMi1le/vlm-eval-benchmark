# Reference runtime used in the archived MiniCPM-V 4.7 AWQ evaluation (amd64).
FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
RUN python3 -m pip uninstall -y opentelemetry-exporter-otlp-common \
    && python3 -m pip install --no-deps transformers==5.18.0 nvidia-nccl-cu13==2.29.7 \
    && python3 -m pip check \
    && if ! command -v python >/dev/null 2>&1; then ln -s "$(command -v python3)" /usr/local/bin/python; fi
