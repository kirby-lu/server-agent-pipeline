#!/bin/bash

docker run -d -it -v ./source:/Projects/source \
 -p ${HOST_PORT}:${SERVER_PORT} \
 --rm --name ${TASK_NAME} \
 --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
 base_image:v1.0 /bin/bash