# CUDA-enabled base image with cuDNN
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    python3-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libx11-6 ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Symlink python3.12 and pip3
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY . .

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Overwrite outdated files in tensorflow package
RUN pip uninstall -y tf-nightly
RUN pip install tf_nightly==2.20.0.dev20250515

EXPOSE 5000

CMD ["python", "main.py"]