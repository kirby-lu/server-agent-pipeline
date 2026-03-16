#!/bin/bash

docker exec \
 -d ${TASK_NAME} \
 sh -c 'export CUDA_VISIBLE_DEVICES=0 && cd /Projects/source && sh run_start_server.sh'