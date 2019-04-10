FROM python:2.7-slim
WORKDIR /model
COPY . /model
RUN pip install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 80
ENV NAME TextAnalyzer
CMD ["python", "main.py"]