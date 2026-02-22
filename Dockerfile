FROM python:3.11-slim

# Instalar curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copiar dependencias primero (mejor cache)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copiar solo lo necesario
COPY app.py .
COPY src/ src/
COPY models/rain_model.pkl models/rain_model.pkl

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]