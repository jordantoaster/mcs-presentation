# mcs-presentation

## Setup

### Docker Setup

1. Make sure you have Docker installed and running on your machine.

2. Start the Ollama service using Docker Compose:
   ```bash
   cd docker
   docker-compose up --build -d
   ```
   This will start the Ollama service on port 11434.

3. You can verify the service is running with:
   ```bash
   docker ps
   ```

### Running the Local UV App

1. Ensure UV is installed on your system

2. Run ```uv run <file_name>.py```