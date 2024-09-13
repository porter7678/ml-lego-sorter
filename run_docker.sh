#!/bin/bash

COMMAND="$@"

# Ensure pigpiod is running on the host
if ! pgrep -x "pigpiod" > /dev/null
then
    echo "Starting pigpiod on host..."
    sudo pigpiod
else
    echo "pigpiod is already running."
fi

# Run Docker container with access to GPIO and camera
docker run --rm --privileged -it \
    --device /dev/video0 \
    -v /dev/gpiomem:/dev/gpiomem \
    -v /dev/mem:/dev/mem \
    -v ~/projects/ml-lego-sorter:/usr/src/app \
    --network host \
    -e PIGPIO_ADDR=localhost \
    -e PIGPIO_PORT=8888 \
    porter7678/lego-image $COMMAND