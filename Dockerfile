# Use the official Python image from the Docker Hub
FROM python:3.11-slim as builder-grap

# Pin the poetry version to avoid breaking changes
ENV POETRY_VERSION=1.8.2
ENV ROOT=/app

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set the working directory
WORKDIR ${ROOT}

# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock ${ROOT}/

# Create README (if necessary)
RUN touch README.md

# Install dependencies without creating a virtual environment
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root

# Copy the entire project to the working directory
COPY . ${ROOT}/

RUN poetry config virtualenvs.create false \
  && poetry install --only main
  
# Set the entry point to your application and include PYTHONPATH
# uvicorn api.main:api --host 0.0.0.0 --port 8000 --reload
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]