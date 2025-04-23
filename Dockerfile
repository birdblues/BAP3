FROM my_miniconda3:latest

# Set the working directory
WORKDIR /workspace

# Install basic build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    git \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the bap3.yml file into the container
COPY bap3.yml .

# Create the conda environment using the bap3.yml file
RUN conda env create -f bap3.yml

# Activate the environment
RUN echo "source activate bap3" > ~/.bashrc

# Set the default command to run when starting the container
CMD ["bash"]