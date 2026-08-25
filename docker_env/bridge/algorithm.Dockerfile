FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY bridge_algorithm_service/requirements.txt /tmp/requirements.txt
RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements.txt

COPY bridge_algorithm_service/ /app/bridge_algorithm_service/

CMD ["uvicorn", "bridge_algorithm_service.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
