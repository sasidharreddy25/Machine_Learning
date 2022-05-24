FROM ubuntu
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt install python3 python3-pip build-essential cmake make -y
RUN pip3 install flask
RUN pip3 install sklearn

COPY ./ /docker-app

ENTRYPOINT /usr/bin/python3 /docker-app/fi/app.py