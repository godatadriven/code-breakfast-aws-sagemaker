FROM mcr.microsoft.com/devcontainers/python:3.10

COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" && \
#     unzip /tmp/awscliv2.zip && \
#     /tmp/aws/install && \
#     rm -rf /tmp/aws && \
#     rm /tmp/awscliv2.zip


RUN sudo apt-get update && sudo apt-get install awscli -y