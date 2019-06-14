FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3.6 python3.6-dev python3-pip python3.6-venv
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["app.py"]

