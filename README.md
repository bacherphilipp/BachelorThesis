# BachelorThesis Filtering Project

This repository contains the code and resources for the BachelorThesis project focused on collaborative and content-based filtering techniques. The project aims to explore and implement different filtering algorithms for recommendation systems.

## Project Structure

- `config/` - Contains configuration files for the project, including hyperparameters and database settings.
- `src/` - The source code for the project:
  - `data_retrieval/` - Scripts for database connection and data retrieval.
  - `models/` - Implementation of various filtering models and helpers.
- `main.py` - The main script to run the project's filtering functions.
- `requirements.txt` - The required libraries and dependencies for the project.

## Installation

To set up the project environment:

1. Clone the repository to your local machine.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies by running:

```shell
pip install -r requirements.txt
```
## Installation

To run the main filtering script, navigate to the project directory and execute:
```shell
python main.py [arguments]
```
Replace [arguments] with the required command-line arguments for the specific filtering technique you want to apply. Use --help to see all available options.

## Configuration

Edit the .ini files in the config/ directory to tweak the hyperparameters and database settings according to your needs.