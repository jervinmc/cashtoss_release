FROM python:3.8-slim

ENV PYTHONUNBUFFERED 1

RUN apt-get -y update && apt-get -y upgrade && apt-get -y install libmariadb-dev-compat libmariadb-dev build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev

WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt                                                                            

EXPOSE 5000

ENTRYPOINT  ["python3"]

CMD ["app.py"]