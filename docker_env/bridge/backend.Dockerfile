FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /backend

COPY backend/requirements.bridge.txt /tmp/requirements.txt
RUN python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements.txt

COPY backend/ /backend/

# 启动前先执行幂等迁移，保证 dvadmin3_celery 等表结构存在
CMD ["sh", "-c", "python manage.py migrate --noinput && uvicorn application.asgi:application --host 0.0.0.0 --port 8002 --workers 2"]
