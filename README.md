# Machine Learning Zoomcamp - 2025 Cohort
This repo holds all of my code related to the 2025 cohort of the Machine Learning Zoomcamp.

## Running homework code
This course utilizes Jupyter notebooks.
I like running things on my local machine, so I set things up to run within docker.
To get started, ensure Docker is installed (Docker desktop is usually the easiest way to set up Docker).
Then run the following command:
```sh
docker compose -f docker/docker-compose.yml up -d homework
```
This will stand up the [homework Jupyter notebooks)[localhost:6592].
Because copying tokens all the time is a headache when I am only intending for this to run locally, I have hard-coded the password to `jupyter`.

## Shutting down
When you are done running this project, you can run the following command to tear down the docker compose setup.
```sh
docker compose -f docker/docker-compose.yml down
```
