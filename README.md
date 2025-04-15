# Neuron Dockerfile

Build the Docker image:
```
$ docker build -t neuron .
```

Run an interactive container
```
$ docker run -v ./workspace:/neuron/workspace -it neuron bash
```

The `-v ./workspace:/neuron/workspace` bind mounts the `workspace` directory
into the container, so changes in the container will be reflected on the host.

When you start the interactive container, `entrypoint` automatically activates the Python virtualenv.
