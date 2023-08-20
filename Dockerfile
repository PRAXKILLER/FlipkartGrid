FROM python:3.11
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
# CMD python app/app.py

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "3000"]

