#works: FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
#FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# System-level deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3-pip \
      python3 \
      python3-dev \
      poppler-utils \
      tesseract-ocr-deu \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y build-essential


# Set python and pip to always point to python3
RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip
 
WORKDIR /workspace

# Set persistent cache directory for Hugging Face models
ENV HF_HOME=/workspace/.cache/huggingface

COPY requirements.txt /workspace/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]