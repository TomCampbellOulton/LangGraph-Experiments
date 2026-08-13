## Following the following guide:
https://www.youtube.com/watch?v=uWLJAtMOVT0

## To run:
# Setup the .env file
First modify all data in it to match your models
Then run the following command: (copies the example .env file and names it .env)
cp .env.example .env

# Under the LLM folder, add 2 folders:
#llama.cpp
The compiled llama cpp for your CUDA versions

#models
This should contain all the models you plan on using (e.g. Qwen 3.6 9b)


# To setup docker: (-d means detached mode so you can close the cmd used to set it up)
docker compose up -d --build

# Once docker is setup up, run the following command to execute main.py
docker compose exec -it agent uv run main.py