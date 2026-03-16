docker exec \
 -d ${task_name} \
 sh -c 'export CUDA_VISIBLE_DEVICES=0 && cd /Projects/${project_name} && sh run_start_server.sh'