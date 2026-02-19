# aig200-ml-model-deployment
**Course**: AIG 2026 - Capstone AIG200 Class, Machine Learning Model Deployment (Individual Assignment).
**Student**: Maminirina Fabien Rakotoson (ID: 117328245)

# Description of the project
This project contains a trained sentiment classifier, returning if a given input is `positive OR neutral OR negative`. It's deployed as a FastAPI service, containerized with Docker, and deployed on AWS LightSail.

# Live testing
Please test the outcome of it by launching this command in CLI, and the text content can be changed as well.
```bash
curl -X POST "https://aig-capstone.jgcz2b6j32c6t.ca-central-1.cs.amazonlightsail.com/predict" -H "Content-Type: application/json" -H "X-API-Key: FABIENRAKOTOSON" -d "{\"text\": \"I was laughing everywhere, that was so fun!\"}"
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
docker run --rm -p 8080:8080 -e API_KEY="FABIENRAKOTOSON" sentiment-api:latest

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
Link to deployed: https://aig-capstone.jgcz2b6j32c6t.ca-central-1.cs.amazonlightsail.com/
The API is deployed on **Amazon LightSail**.