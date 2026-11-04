# aig200-ml-model-deployment
**Course**: AIG 2026 - Capstone AIG200 Class, Machine Learning Model Deployment (Individual Assignment).
**Author**: Maminirina Fabien Rakotoson

# Description of the project
This project contains a trained sentiment classifier, returning if a given input is `positive OR neutral OR negative`. It's deployed as a FastAPI service, containerized with Docker, and deployed on AWS LightSail.

# Testing the API
The public demo deployment has been taken down. Run it locally (see below), then call it with the API key you set in `API_KEY`:
```bash
curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -H "X-API-Key: <your-api-key>" -d "{\"text\": \"I was laughing everywhere, that was so fun!\"}"
```

# Datasets
- Kaggle Dataset link: https://www.kaggle.com/datasets/mabubakrsiddiq/sentiment-analysis-dataset-31k-samples
- It is stored in the repository as `sentiment_dataset.csv` as well

# Model approach selection
### 1st model (TF-IDF + Logistic Regression)
- Saved as: `artifacts/sklearn_sentiment.joblib`

### 2nd model (Fine-tuned DistilBERT from HuggingFace) <- Chosen for deployment
- This model is the one we use for the deployed API
- Base model: `distilbert-base-uncased`
- Saved as:
  - model + tokenizer: `artifacts/transformer/`
  - label mapping: `artifacts/label_map.json`

# Local Run + deployment 
#### Run the main.py locally and test the end-to-end outputs of the model
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080

#### Docker commands
docker build -t sentiment-api:latest .
docker run --rm -p 8080:8080 -e API_KEY="<your-api-key>" sentiment-api:latest

#### Tag the local instance, then deploy it to Docker Hub
docker tag sentiment-api:latest rakotosonfabien/sentiment-api:latest
docker login
docker push rakotosonfabien/sentiment-api:latest

#### Set up on Amazon LightSail
- Went to AWS LightSail
- Created a new container
- Specified the deployment information for the docker image
- Specified the API_KEY, port,health check path

# Deployment information
The API was deployed on **Amazon LightSail** (container service) for the course; that demo is no longer online.