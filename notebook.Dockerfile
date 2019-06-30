
FROM jupyter/base-notebook:python-3.7.3

ENV PASSWORD password
ENV PYTHONPATH "${PYTHONPATH}:$HOME/src"

# Time for some black magic. Merging Conda with Pipenv
RUN pip install pipenv
COPY ./Pipfile ./Pipfile.lock ./
RUN pipenv lock -r | sed -n '1!p' > requirements.txt
RUN pip install -r requirements.txt --ignore-installed
RUN pipenv --rm
RUN cat ./requirements.txt

COPY . .