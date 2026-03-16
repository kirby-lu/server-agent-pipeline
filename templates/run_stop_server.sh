#!/bin/bash

docker exec \
 -d ${TASK_NAME} \
 sh -c 'cd /Projects/source && sh run_stop_server.sh'