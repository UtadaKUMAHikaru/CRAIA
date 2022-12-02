# FROM python:3.10
FROM clearlinux/numpy-mp
FROM tensorflow/tensorflow
# FROM tiangolo/uvicorn-gunicorn-fastapi

# COPY ./app /app

COPY . /code

RUN pip install --no-cache-dir --upgrade -r /code/app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /code/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

# WORKDIR /craia_app/app

# # Configure Poetry
# ENV POETRY_VERSION=1.2.0
# ENV POETRY_HOME=/opt/poetry
# ENV POETRY_VENV=/opt/poetry-venv
# ENV POETRY_CACHE_DIR=/opt/.cache

# # Install poetry separated from system interpreter
# RUN python3 -m venv $POETRY_VENV \
#     && $POETRY_VENV/bin/pip install -U pip setuptools \
#     && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# # Add `poetry` to PATH
# ENV PATH="${PATH}:${POETRY_VENV}/bin"

# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# RUN python3 -m pip install --user pipx
# RUN python3 -m pipx ensurepath
# # RUN /bin/bash
# RUN ~/.local/bin/pipx install poetry==1.2.0
# RUN poetry install

# EXPOSE 8000

# ENTRYPOINT ["python"]

# CMD ["main.py"]