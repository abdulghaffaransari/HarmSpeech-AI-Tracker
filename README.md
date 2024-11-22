
---

# **HarmSpeech-AI-Tracker**  
### *Empowering Responsible Communication through AI*

---

## **Project Description**  
HarmSpeech-AI-Tracker is a robust AI-powered system designed to identify and analyze hate speech and offensive language. Using cutting-edge machine learning and NLP techniques, this project enables effective content moderation, fosters healthier communication, and promotes a safer online environment.

---

## **Key Features**  
- **Advanced NLP Models**: Leveraging TensorFlow and Scikit-learn for accurate hate speech detection.  
- **Data Pipeline Automation**: Efficient ingestion, transformation, and preprocessing of data.  
- **Real-Time API**: FastAPI integration for seamless prediction and deployment.  
- **Scalable Architecture**: Modular design with Docker support for easy scalability.  
- **Cloud Ready**: Sync capabilities with Google Cloud Storage for production-level deployment.

---

## **Project Structure**  
The project is organized into multiple modules for easy maintenance and scalability:  
```
hate/
    components/              # Core modules for data and model handling
        data_ingestion.py
        data_transformation.py
        model_trainer.py
        model_evaluation.py
        model_pusher.py
    configuration/           # Configuration and cloud sync utilities
    constants/               # Constants for configuration and paths
    entity/                  # Data and configuration entities
    exception/               # Custom exception handling
    logger/                  # Logging utilities
    pipeline/                # Training and prediction pipelines
    ml/                      # Model implementation
app.py                       # FastAPI application for predictions
demo.py                      # Demo script for testing
requirements.txt             # Dependencies for the project
Dockerfile                   # Containerization configuration
setup.py                     # Package setup for installation
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

Access the API docs at: `http://127.0.0.1:8000/docs`

---

## **Current Progress**  
- ✅ Project structure initialized.  
- ✅ Dependencies installed.  
- ✅ Data pipeline modules are in progress.  
- 🔄 Model training in progress.  
- ⏳ API and deployment integration (coming soon).  

---

## **Technologies Used**  
- **Programming Language**: Python  
- **Libraries**: TensorFlow, Pandas, Scikit-learn, Matplotlib, Seaborn, NLTK  
- **Frameworks**: FastAPI  
- **Tools**: Docker, Google Cloud Storage  

---

## **Future Enhancements**  
- Enhance the model with transformers for better accuracy.  
- Integrate cloud deployment with Kubernetes.  
- Add real-time content moderation features.  

---

## **Author**  
**Abdul Ghaffar Ansari**  
📧 [abdulghaffaransari9@gmail.com](mailto:abdulghaffaransari9@gmail.com)  

---