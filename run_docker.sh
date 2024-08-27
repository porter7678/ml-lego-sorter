#!/bin/bash

COMMAND="$@"

docker run --rm -it --device /dev/video0 -v ~/projects/ml-lego-sorter:/usr/src/app porter7678/lego-image $COMMAND
