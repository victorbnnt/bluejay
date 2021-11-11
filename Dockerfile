FROM tensorflow/tensorflow

RUN apt-get update
RUN python -m pip install --upgrade pip

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

EXPOSE 8080

RUN ls -la $APP_HOME/

# Install dependencies
RUN pip install -r requirements.txt

CMD ["python3", "bluejay.py"]
