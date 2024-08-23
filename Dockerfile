FROM python:3.9

RUN python3 --version

WORKDIR /usr/src/app

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install scikit-learn matplotlib

RUN echo "porter complete."