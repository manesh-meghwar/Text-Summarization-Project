# this file is just create directories nad files for this project automatically using this command => python template.py

import os
from pathlib import Path
import logging          # It logs all information

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:') 


project_name = "textSummarizer"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",              # local package and here __init__.py shows the folder as a packages
    f"src/{project_name}/conponents/__init__.py",   # so every folder looks like a package
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",                       # all project's configuration here
    "params.yaml",                              # all project's parameters here
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",                                 # It helps to setup local packages
    "research/trials.ipynb",                    # Here we add research notebooks

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")    
    else:
        logging.info(f"{filename} is already exists")