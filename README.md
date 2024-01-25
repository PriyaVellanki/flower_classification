# Flower Classification

## Problem Statement

Botanical gardens and flower enthusiasts often encounter difficulties in manually identifying various flower species, especially when dealing with a large number of species. Automated flower classification aided by machine learning can significantly aid in the identification process and improve efficiency. This project showcases a simple use case of classifying if a given image of a flower is a Daisy or a Dandelion. This can easily be extended to scale to many flower categories.

## About the dataset

Used the `Kaggle` [flower classification data set](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition/data), which has been collected from scraped data from `flickr`, `Google`, and `Yandex` images.

## Deliverables

Machine Learning:
- Utilized the capabilities of state-of-the-art architecture provided by `Keras`. 
- Employed the transfer learning technique to construct two distinct models based on Inception and MobileNet.

Tuning:
- Trained with various learning rates, added additional inner layers, regularization and augmentation.

Model Evaluation:  
- After thorough evaluation, selected MobileNet as the final model due to its compactness and efficiency, with comparable accuracy to Inception but with fewer parameters.

Ease of Use:
- Containerized using Docker and serviced by REST APIs using Flask.

Cloud Deployment:
- Deployed model to AWS Lambda.

User Friendly Interface: 
- Web interface deployed as `streamlit` web app.

## How to use

### Clone the repo
```
git clone https://github.com/PriyaVellanki/flower_classification.git
```
### Repository structure (description)

- flower_classification (root)
    - app (Flask API service)
    - serverless (AWS Lambda Deployment and Docker File)
    - model (final model)
    - data (Jupyter Notebook files and env files)

### Setup local environment

```
pip install pipenv
pipenv --python 3.7
pipenv install 
pipenv shell

# install tflite-runtime
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

### Running the train notebook

GPU is needed to run the notebook. One can run on `Google Collab` and any other platform which supports GPU. It does save models multiple models from `Mobilenet` and `Xcpetion`. Make sure to delete unwanted files to make sure disk is not getting full.

### How to run app locally (without Docker and with Flask)

1. Clone the repo
2. pipenv install
3. pipenv shell
4. pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
5. cd app/
6. Run python3 app/predict.py  
7. In another window, run the test_api.py

You should see the result similar to below:
```python
python3 test.py
{'daisy': 0.9988583326339722, 'dandelion': -4.313272953033447}
```

### How to run app locally (with Docker)

- Change to serverless directory. `cd flower_classification/serverless`
- `docker build --platform linux/amd64 -t flower-classifcation:v1 .`
- `docker run --platform linux/amd64 --rm -p 9000:8080 flower-classifcation:v1`
- Make sure `docker container` is running. You can check this by running `docker ps`.
- Run the `python3 test_docker_local.py` or `curl "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"url":"https://github.com/PriyaVellanki/flower_classification/raw/main/data/11124324295_503f3a0804.jpg"}'`

Expected result:

```python
flower_classification % curl "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"url":"https://github.com/PriyaVellanki/flower_classification/raw/main/data/11124324295_503f3a0804.jpg"}'
{"daisy": 0.9988558292388916, "dandelion": -4.313270568847656}
```

## Cloud deployment
Hosted on `AWS Lambda` and `Streamlit` Cloud 

### AWS Lamda

AWS Lambda Setup Instructions:
Official instructions for reference: https://docs.aws.amazon.com/lambda/latest/dg/python-image.html#python-image-instructions

- Build, tag and run docker container image : `docker tag flower_classification:latest ${REMOTE_URI}` 
- Login to AWS ECR: `aws ecr get-login-password | docker login --username AWS --password-stdin <account_url>`
- Create AWS elastic container registry (ECR) repository to store the image : `aws ecr create-repository --repository-name <repo_name>`
- Publish docker container image to ECR repository as tagged image : `docker push ${REMOTE_URI}`
- Create, configure and test AWS Lambda function
- Create, configure and test AWS Rest API Gateway to access Lambda function
- Make prediction using POST METHOD /predict

Run the following command to test: 

```
curl 'https://ivw7r2k600.execute-api.us-west-1.amazonaws.com/stage/predict' -d '{"url":"https://github.com/PriyaVellanki/flower_classification/raw/main/data/11124324295_503f3a0804.jpg"}'

```

Expected output:

<img width="1440" alt="Screenshot 2024-01-24 at 8 09 50 PM" src="https://github.com/PriyaVellanki/flower_classification/assets/36514922/e1116dc4-3ed8-4eb2-a7db-87e692bfc8cd">

### Streamlit Cloud
App url : https://flowerclassification-daisyordandelion.streamlit.app/

`Streamlit` is an open-source Python framework for machine learning and data science teams.

- Open the App url in the browser: `https://flowerclassification-daisyordandelion.streamlit.app/`
-  Download the test data from `https://github.com/PriyaVellanki/flower_classification/raw/main/data/11124324295_503f3a0804.jpg` and upload image in the web app
- Click on `Predict` to see the result

Streamlit webapp result example:

<img width="1000" alt="Screenshot 2024-01-21 at 11 16 12 PM" src="https://github.com/PriyaVellanki/flower_classification/assets/36514922/d90628ab-b427-40dc-88e8-191e957e3923">
