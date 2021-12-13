# Project Template

The focus of this project is to demonstrate how to create a custom algorithm which runs in AWS Sagemaker.
At the end of this process, we should be able to create a container with reproducible code to be used as a source
for predictions in production.

Regarding the machine learning problem,
it's a regression type problem that aims to estimates the **SalePrice** of a house.

### Data Dependencies
We are using the following Data Source:

| Source | Description |
|--------|-------------|
|[Kaggle's House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)| Info about a number of variables describing (almost) every aspect of residential homes in Ames, Iowa.|------------|

## Repository Structure
- ```.github```: Runs Github Actions' CI/CD for checking the project integrity.

- ```Security```: Zup's security resources. 

- ```analysis```:  Local for Jupyter Notebook demonstration on how the fetched data was used to train a machine 
learning model.

- ```houses_regression```: The project modules.

- ```tests```: The project's test directory.

## Installation
Run the commandss below on the root directory to install the proper dependencies.

```
pip install .
```

For a functional replicability, it's necessary to set your **AWS_ACCESS_KEY_ID**, **AWS_SECRET_ACCESS_KEY**, 
**AWS_SESSION_TOKEN** and **REGION_NAME** as environment variables.

```
export AWS_ACCESS_KEY_ID="anaccesskey"
export AWS_SECRET_ACCESS_KEY="asecretkey"
export AWS_SESSION_TOKEN="asessiontoken"
export AWS_DEFAULT_REGION="yourregion"
```

## Usage

### Testing the project
For testing, do run the command below from the root directory:

```
tox -e test_package
```

For linting, do run the command below from the root directory:

```
tox -e lint && tox -e typechecks
```

### Docker

Use ```docker build -t <IMAGE_NAME> .``` (you may need to use ```sudo``` before the command)
to build the container. After that you can check if the project is ready to be pushed to AWS ECR running the train 
and serve commands.

The train command runs the **train** script inside the project, running a training job. If everything is right
you should see the model metrics as an output.

The serve method ignites a gunicorn server to communicate with other services. It's essential for AWS Sagemaker's API
to use the container in AWS ECR as a source training job.

Run the commands below:

```
docker run <IMAGE_NAME> train
docker run <IMAGE_NAME> serve
```
