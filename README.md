
---

# **HarmSpeech-AI-Tracker**  
**Empowering Responsible Communication through AI**

---

## **Project Description**
**HarmSpeech-AI-Tracker** is a state-of-the-art AI-powered platform designed to detect and analyze hate speech and offensive language. It combines advanced **Natural Language Processing (NLP)** and machine learning techniques with a robust software architecture, enabling real-time predictions and content moderation. By fostering healthier communication, this project addresses critical issues in online interactions and promotes a safer, inclusive environment.

The system employs scalable cloud solutions, **CI/CD pipelines**, and automated workflows, making it an ideal choice for production-level deployment in fast-paced software ecosystems.

---

## **Key Features**
- **Advanced NLP Models**: Precise detection of hate speech using **TensorFlow** and **Scikit-learn**.  
- **GCP Model Management**: Models are stored in **Google Cloud Storage (GCS)** buckets, ensuring efficient version control and retraining workflows.  
    - Automatically retrieves the **latest model** from GCP when a new model is trained.  
    - Compares the accuracy of the existing model with the new one.  
    - Saves the **best-performing model** back to the GCP bucket for production use.  
- **Test-Driven Development**: Comprehensive unit tests for every module to ensure reliability and quality.  
- **CI/CD Pipelines**: Seamless integration with **CircleCI** for automated testing, building, and deployment.  
- **AWS Deployment**: Transitioned from GCP to **AWS EC2** and **S3**, ensuring scalable and robust production deployment.  
- **Real-Time API**: FastAPI-based endpoints for quick and accurate predictions.  
- **Data Pipeline Automation**: Fully automated data ingestion, transformation, and preprocessing workflows.  
- **Scalable Architecture**: Dockerized components with modular design for ease of deployment and maintenance.  

---

## **Project Architecture**
The project follows a clean, modular architecture designed for scalability and maintainability:

```
HarmSpeech-AI-Tracker/
│
├── hate/
│   ├── components/              # Core modules for data and model handling
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   ├── model_pusher.py
│   ├── configuration/           # Configuration and cloud sync utilities
│   ├── constants/               # Constants for configuration and paths
│   ├── entity/                  # Data and configuration entities
│   ├── exception/               # Custom exception handling
│   ├── logger/                  # Logging utilities
│   ├── pipeline/                # Training and prediction pipelines
│   ├── ml/                      # Model implementation
│
├── tests/                       # Unit tests for all modules
│   ├── test_0_data_ingestion.py
│   ├── test_1_data_transformation.py
│   ├── test_2_model_training.py
│   ├── test_3_model_evaluation.py
│   ├── test_4_model_pusher.py
│
├── app.py                       # FastAPI application for predictions
├── demo.py                      # Demo script for testing
├── requirements.txt             # Dependencies for the project
├── Dockerfile                   # Containerization configuration
├── .circleci/                   # CI/CD pipeline configurations
│   └── config.yml
├── setup.py                     # Package setup for installation
```

---

## **Installation and Usage**

### **1. Clone the Repository**
```bash
git clone https://github.com/abdulghaffaransari/HarmSpeech-AI-Tracker.git
cd HarmSpeech-AI-Tracker
```

### **2. Set Up Environment**
Create a Conda environment and install dependencies:
```bash
conda create --name hate python=3.10
conda activate hate
pip install -r requirements.txt
```

### **3. Run the Application**
Start the FastAPI server for predictions:
```bash
uvicorn app:app --reload
```
Access the API documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  

### **4. Run Tests**
Execute test cases to validate the system:
```bash
pytest tests/
```

---

## **GCP Model Management Workflow**
The **HarmSpeech-AI-Tracker** incorporates an efficient **GCP-based model management system** to maintain the **best-performing model** for production:  

1. **Model Storage**:
   - All trained models are stored in **Google Cloud Storage (GCS)** buckets, ensuring a centralized repository for version control and accessibility.
   - Model artifacts include:
     - Trained model file (`model.h5`)
     - Associated metadata for evaluation metrics (e.g., accuracy, precision).

2. **Model Comparison**:
   - During the training process, the system retrieves the **latest model** from GCP for evaluation.  
   - The newly trained model is compared against the current production model using predefined metrics (e.g., accuracy, F1-score).  

3. **Model Deployment**:
   - If the new model outperforms the current model, it is uploaded back to the GCP bucket and marked as the **production-ready model**.  
   - This ensures that only the best model is used in deployment, maintaining high performance and reliability.

4. **Cloud Sync**:
   - Automatic syncing of model artifacts between **GCP buckets** and the **AWS EC2** production environment.  

---

## **Test Cases**
**Test-Driven Development (TDD)** is a cornerstone of the project. Each module is rigorously tested to ensure reliability and robustness:

- **`test_0_data_ingestion.py`**: Verifies that the data ingestion pipeline reads, validates, and stores raw data correctly.  
- **`test_1_data_transformation.py`**: Tests the preprocessing steps, ensuring consistency and correctness in data transformation.  
- **`test_2_model_training.py`**: Evaluates the model training process, including artifact generation and performance metrics.  
- **`test_3_model_evaluation.py`**: Assesses the evaluation logic, ensuring accurate comparison between models.  
- **`test_4_model_pusher.py`**: Validates the process of pushing models to the cloud (GCP bucket) or production environments.  

These test cases enable developers to debug and optimize the system efficiently, ensuring production-readiness.

---

## **Deployment**
### **AWS Cloud Deployment**
The project has been successfully deployed on **AWS**, leveraging the following services:
- **EC2**: Hosting the FastAPI application for real-time predictions.  
- **S3**: Storing intermediate data and model artifacts for scalability.  
- **RDS**: Storing metadata for model evaluation and API logs.  
- **CircleCI**: Automated CI/CD pipelines for continuous testing, building, and deployment.

### **GCP Integration**
The initial deployment utilized **GCP** with seamless integration for:
- **Google Cloud Storage**: Centralized storage for model artifacts and evaluation logs.
- **Model Comparison**: Ensures that the production model is always the best-performing version.  
- **Retraining Workflow**: Automatically integrates the latest model into the pipeline for periodic updates.

---

## **Technologies Used**
- **Programming Language**: Python  
- **Libraries**: TensorFlow, Scikit-learn, Pandas, Matplotlib, Seaborn, NLTK  
- **Frameworks**: FastAPI  
- **DevOps Tools**: Docker, CircleCI, Google Cloud Storage (GCS), AWS (EC2, S3, RDS)  
- **Databases**: AWS RDS  

---

## **Future Enhancements**
- Incorporate **Transformer-based models** like **BERT** or **RoBERTa** for enhanced accuracy.  
- Add **multi-language support** for detecting hate speech globally.  
- Real-time content moderation with **WebSocket-based communication**.  
- Transition to **Kubernetes** for scalable cloud orchestration.  

---

## **Author**
**Abdul Ghaffar Ansari**  
📧 abdulghaffaransari9@gmail.com  

--- 
