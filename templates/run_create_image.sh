docker run -d -it -v ./${project_name}:/Projects/${project_name} \
 -p ${host_port}:${server_port} \
 --rm --name ${task_name} \
 --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
 base_image:v1.0 /bin/bash