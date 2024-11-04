# dnq-supermario

Train a Deep Q Learning (DQN) agent to play Super Mario.

## Project structure

## Local development

### Type anotation

Install VS Code Extension `Pylance` from Microsoft and set typeCheckingMode in settings to `strict`.

### Dev Container

In order to provide a uniform dev environment with the correct package versions, dev container is one of the best solutions.
Setup:

- Use VS Code as IDE
- Install extension "Dev Containers" (Microsoft)
- Start docker
- Run command (F1) "Dev Containers: Rebuild and Reopen in Container"
- Note: Before pushing to git, you should first close the container via command (F1) "Dev Containers: Reopen Folder locally"

### Virtual environment

An alternative for Dev Container is virtual environment, with which you only have to install the required packages once.

#### Windows

- `python -m venv venv`
- `.\venv\Scripts\activate`
- `cd .\venv\`
- `py.\venv\Scripts\activate -3.12 -m venv .`
- `cd ..`
- `pip install -r .\requirments.txt`

#### Linux (22.04 LST)

- `sudo apt install python3.12-venv`
- `python3.12 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `pip install ipykernel`
- `python -m ipykernel install --user`

## Run tests with VS code

1. Install `Python Test Explorer for Visual Studio Code` extension from `Little Fox Team`
2. If the no build error, the tests will be recogniyed an can be then run locally.

## Train and validate the models

### Breakout

- To train the model to play Breakout, run `python3 deep_q_breakout.py -c config/breakout.yaml`
- To validate the model, run `python3 deep_q_breakout_validation.py`
