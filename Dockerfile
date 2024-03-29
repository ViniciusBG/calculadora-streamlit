FROM python:3.8-slim-buster
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 8501

CMD streamlit run app.py --server.port 8501
