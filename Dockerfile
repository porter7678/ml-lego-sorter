FROM python:3.9

# Show Python version
RUN python3 --version

# Set the working directory inside the container
WORKDIR /usr/src/app

# Install dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install Python packages
RUN pip3 install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install scikit-learn matplotlib opencv-python
RUN pip3 install pigpio gpiozero

# Indicate that the build process is complete
RUN echo "porter complete."
