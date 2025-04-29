# 使用官方 Python 3.10 轻量镜像
FROM python:3.10-slim

# 禁用输出缓存，方便日志查看
ENV PYTHONUNBUFFERED=1

# 设置工作目录
WORKDIR /app

# 复制依赖列表并安装
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制你的 Streamlit 应用
COPY app.py ./

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
