FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "disfluency_env", "/bin/bash", "-c"]

COPY . .

EXPOSE 8000

CMD ["conda", "run", "-n", "disfluency_env", "python", "Inference/server.py"]
