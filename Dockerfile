FROM python:3.11.4

WORKDIR D:\Workspaces\VSCode Workspace\ML\app
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y \
    python3-tk \
    xauth \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENV FLASK_APP=app.py

EXPOSE 5000

# CMD ["flask", "run", "--host=0.0.0.0"]

# Thiết lập biến môi trường DISPLAY
# ENV DISPLAY=0:0
CMD ["python", "./app.py"]
