FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# requirements.txt est à la racine du contexte (.)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copier les fichiers clients depuis le dossier clients/ du contexte
COPY clients/client1.py ./client1.py
COPY clients/client2.py ./client2.py
COPY clients/client3.py ./client3.py