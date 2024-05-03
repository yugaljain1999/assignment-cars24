FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./utils /code/utils
COPY ./model /code/model

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]