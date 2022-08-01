# ZOLO project
ZOLO provide nano credit via a mobile app.
The project goal is to make a machine learning model to avoid payment defaults

## links

* MLFLOW: https://zolo-app.herokuapp.com/#/experiments/3

## Details for certification purpose

* email adress: ghazibouchnak@gmail.com
* video link: 

### Data

* For confidential reasons the datasets can't be shared

### Prerequisites

The source code is written in Python 3  
The python packages can be installed with pip : pip3 install or !pip install if in jupyter notebook


## Deployment

* Run MLFLOW:  
docker run -it -v "$(pwd):/home/app" -e PORT=80 -e AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY" -e AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY" -e BACKEND_STORE_URI="YOUR_BACKEND_STORE_URI" -e ARTIFACT_ROOT="YOUR_S3_BUCKET" -e MLFLOW_TRACKING_URI="YOUR_MLFLOW_HEROKU_SERVER_URI" YOUR_IMAGE_NAME python app2.py  

* Create your app and deploy it on HEROKU:  
1-heroku container:login  
2-heroku create YOUR_APP_NAME  
3-heroku container:push web -a YOUR_APP_NAME  
4-heroku container:release web -a YOUR_APP_NAME  
5-heroku open -a YOUR_APP_NAME  

## Built With

* ZOLO_Project_PREPROCESSING.ipynb: Data preprocessing  
* app2.py: MLFLOW app  
* graph_scores_v2.ipynb: Feature importance and scores  

## Authors

**Ghazi BOUCHNAK** - [Ghazi-13](https://github.com/Ghazi-13)


