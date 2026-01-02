FROM python:3.14-slim

WORKDIR /app

COPY models .
COPY samples .
COPY seeds .

COPY main.py .
COPY generator.py .

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gunicorn

EXPOSE 5000
CMD [ "gunicorn", "-b", ":5000", "--access-logfile", "-", "--error-logfile", "-", "main:app" ]