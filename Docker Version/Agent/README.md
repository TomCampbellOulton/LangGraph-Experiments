## Following the following guide:
https://www.youtube.com/watch?v=uWLJAtMOVT0

## To run:
# 

# To setup docker: (-d means detached mode so you can close the cmd used to set it up)
docker compose up -d --build

# Once docker is setup up, run the following command to execute main.py
docker compose exec -it agent uv run main.py