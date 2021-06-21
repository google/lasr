### Build with docker
To access GPU during docker build (credits to [this](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime) thread):
```
sudo apt-get install nvidia-container-runtime
```
Edit/create the /etc/docker/daemon.json with content:
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         }
    },
    "default-runtime": "nvidia"
}
```
Restart docker daemon:
```
sudo systemctl restart docker
```
Build docker image:
```
sudo docker build --tag lasr:latest -f docker/Dockerfile ./
```
Note to run commands in docker container,
```
docker run -v $(pwd):/lasr --gpus all lasr bash -c 'cd lasr; source activate lasr; your command'
```
