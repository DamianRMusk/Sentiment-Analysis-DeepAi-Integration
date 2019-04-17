FROM tensorflow/tensorflow:1.12.0-gpu

WORKDIR /model
COPY . /model

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["python", "entrypoint.py"]

