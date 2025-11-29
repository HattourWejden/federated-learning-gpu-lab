FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# requirements.txt est à la racine du contexte (.)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copier le fichier aggregator.py depuis le dossier agregator/ du contexte
COPY agregator/aggregator.py ./aggregator.py