# flower_classification

## Problem Statement

Botanical gardens and flower enthusiasts often encounter difficulties in manually identifying various flower species, especially when dealing with a large number of species. Automated flower classification using machine learning can significantly aid in the identification process. This project is to identify whether the given image is Daisy or Dandelion. In future, would like to add more flower categories

## About the Dataset

Used kaggle Flower classification data set which is collected from scraped data from flickr, google images, and yandex images.
Link to Original Datset : https://www.kaggle.com/datasets/alxmamaev/flowers-recognition/data

## Goals/Key Deliverables

Machine Learning Model:
Utilized the capabilities of state-of-the-art architectures provided by Keras. Employed the transfer learning technique to construct two distinct models based on Inception and MobileNet.
Tuning:
Trained with various learning rates, added additional inner layers ,Regularization and augmentation.

Model Evaluation:  After thorough evaluation, selected MobileNet as the final model due to its compactness and efficiency, with comparable accuracy to Inception but with fewer parameters.
User Friendly Interface : Deployed model to AWS lambda as well streamlt web app



# How to Use

#### Clone the Repo
```python
git clone https://github.com/PriyaVellanki/flower_classification.git
```
#### Repository Structure
```python
--`Flower_classification`-- Root Folder
| 
|___app - Flask API Service 
|
|_____serverless - AWS Lambda Deployment and Docker File
|
|_______models -- Final Model
|
|__data -- Test Data
|_______________Jupyter Notebook files and env files


```
#### Setup Local Environment
```python
pip install pipenv
pipenv --python 3.7
pipenv install 
pipenv shell

# install tflite-runtime
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

#### Running Train Notebook

GPU is needed to run the notebook. One can run on Google Collab and any other platform which supports GPU. It does save models multiple models from Mobilenet and Xcpetion. Make sure to delete which are needed to make sure disk is not getting full

#### How to run app locally[without Docker and With Flask]
```python
1.Clone the repo
2.pipenv install
3.pipenv shell
4.pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
5.cd app/
6.Run python3 app/predict.py  
7.In another window, run the test_api.py
```
You should see the result similar to below:


#### How to run app locally with Docker

```python
1.cd flower_classification/serverless
2.docker build --platform linux/amd64 -t flower-classifcation:v1 . 
3.docker run --platform linux/amd64 --rm -p 9000:8080 flower-classifcation:v1
4.Make sure docker is running. You can check docker status by running docker ps.
5.Run the python3 test_docker_local.py or curl "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"url":"https://github.com/PriyaVellanki/flower_classification/raw/main/data/11124324295_503f3a0804.jpg"}'

```

#### Cloud Deployment
Hosted on AWS lambda and Streamlit Cloud 


AWS Lambda Setup Instructions:

Instructions Link : https://docs.aws.amazon.com/lambda/latest/dg/python-image.html#python-image-instructions

Build, tag and run docker container image : docker tag flower_classification:latest ${REMOTE_URI} 
Login to AWS ECR: aws ecr get-login-password | docker login --username AWS --password-stdin <account_url>
Create AWS elastic container registry (ECR) repository to store the image : aws ecr create-repository --repository-name <repo_name>
Publish docker container image to ECR repository as tagged image : docker push ${REMOTE_URI}
Create, configure and test AWS Lambda function
Create, configure and test AWS Rest API Gateway to access Lambda function
Make prediction using POST METHOD /predict

Run the following command to test : 
```python

curl 'https://ivw7r2k600.execute-api.us-west-1.amazonaws.com/stage/predict' -d '{"url":"https://github.com/PriyaVellanki/flower_classification/raw/main/data/11124324295_503f3a0804.jpg"}'

```


#### Test Result

```python
(flower_classification) (base) kishore@Kishores-MacBook-Air app % curl 'https://ivw7r2k600.execute-api.us-west-1.amazonaws.com/stage/predict' -d '{"url":"https://github.com/PriyaVellanki/flower_classification/raw/main/data/11124324295_503f3a0804.jpg"}' 
{"daisy": 0.9988583326339722, "dandelion": -4.313272953033447}

```

#### Streamlit Cloud
App url : https://flowerclassification-daisyordandelion.streamlit.app/

##### How to Use/Test:
Streamlit is an open-source Python framework for machine learning and data science teams.

```python
1.Open the App url in the browser: https://flowerclassification-daisyordandelion.streamlit.app/
2.Download the test data from https://github.com/PriyaVellanki/flower_classification/raw/main/data/11124324295_503f3a0804.jpg and upload image in the web app
3. Click on Predict to see the result

```

#### Streamli webapp Results

<img width="1000" alt="Screenshot 2024-01-21 at 11 16 12 PM" src="https://github.com/PriyaVellanki/flower_classification/assets/36514922/d90628ab-b427-40dc-88e8-191e957e3923">












