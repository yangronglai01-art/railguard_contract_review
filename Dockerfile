# 使用项目指定的Python 3.11运行环境。
FROM python:3.11-slim

# 禁止生成.pyc文件，立即输出日志，并关闭pip版本提示。
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# 后续复制、安装和启动命令均在/app目录执行。
WORKDIR /app

# 创建固定UID的非root运行用户，降低容器权限。
RUN groupadd --system --gid 10001 railguard \
    && useradd \
        --system \
        --uid 10001 \
        --gid railguard \
        --create-home \
        --home-dir /home/railguard \
        railguard

# 先复制依赖描述文件和源码，利用Docker构建缓存。
COPY pyproject.toml README.md ./
COPY src ./src

# 安装RailGuard及其运行依赖。
RUN python -m pip install .

# 复制演示知识库和合同文件。
COPY --chown=railguard:railguard data/demo ./data/demo

# 创建SQLite运行目录，并允许非root用户写入。
RUN mkdir -p /app/data/runtime \
    && chown -R railguard:railguard \
        /app \
        /home/railguard

# 后端和前端进程均使用非root用户运行。
USER railguard

# 声明FastAPI和Streamlit使用的容器端口。
EXPOSE 8000 8501

# 默认启动FastAPI；compose中的前端服务会覆盖该命令。
CMD ["python", "-m", "uvicorn", "railguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]