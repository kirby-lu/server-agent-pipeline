docker exec \
 -d ${task_name} \
 sh -c 'cd /Projects/${project_name} && sh run_stop_server.sh'