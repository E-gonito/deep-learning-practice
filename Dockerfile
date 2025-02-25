#---------------------------------------------
# Choose a base image appropriate for your needs.
# - For CPU-only: tensorflow/tensorflow:latest
# - For GPU support: tensorflow/tensorflow:latest-gpu
#---------------------------------------------
    FROM tensorflow/tensorflow:latest-gpu

    # Switch to root user if needed (the base image might already be root).
    USER root
    
    # Update pip and install additional Python packages (NumPy, pydot, etc.)
    RUN apt-get update && apt-get install -y \
        graphviz \
        && rm -rf /var/lib/apt/lists/* \
        && pip install --no-cache-dir --upgrade pip numpy pydot
    
    # Create a working directory (optional but recommended).
    WORKDIR /app
    
    # Copy your local files into the container.
    # (If your model script is called "app.py", for example.)
    COPY . /app
    
    # By default, the container will simply start a shell.
    # To run your Python script automatically, set an entrypoint or command.
    # Uncomment to run a specific script at container startup.
    # CMD ["python", "app.py"]
    
    #---------------------------------------------
    # End of Dockerfile
    #---------------------------------------------