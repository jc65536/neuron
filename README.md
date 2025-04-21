# Neuron Dockerfile

### Build the Docker image:
```
$ docker build -t neuron .
```

### Run an interactive container
```
$ docker run -v ./workspace:/neuron/workspace -it neuron bash
```

The `-v ./workspace:/neuron/workspace` bind mounts the `workspace` directory
into the container, so changes in the container will be reflected on the host.

When you start the interactive container, `entrypoint` automatically activates
the Python virtualenv, so you don't need to do so every time you run the
container.

### Attach VS Code to container

Install the Docker extension, then in the left sidebar right click the neuron
container. If the container is stopped, start it. Once it's running, select
"Attach Visual Studio Code". This will open a VS Code instance in the container.
If you used the command from above, it will automatically apply the bind mount,
so you can open directory `/neuron/workspace` to see the Python files. For
library imports to work, you may need to tell VS Code to find the virtualenv,
which is located in `/neuron/venv`.
