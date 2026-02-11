# 🌾 SmartAgroCare AI – Intelligent Crop Monitoring System

SmartAgroCare AI is an end-to-end **AI-based smart agriculture decision support system** designed to help farmers and agricultural stakeholders detect plant diseases, recommend suitable crops and fertilizers, and interact with an intelligent AI assistant for expert guidance.

The system integrates **Deep Learning, Machine Learning, and Generative AI** to provide accurate, real-time, and multilingual agricultural insights.

---

## 🚀 Key Features

### 🌿 Plant Disease Detection
- Uses a **Convolutional Neural Network (CNN)** to detect plant diseases from leaf images
- Displays predicted disease name with confidence score

### 🌱 Crop Recommendation System
- Machine Learning–based recommendation using **Random Forest**
- Considers soil nutrients (N, P, K), temperature, humidity, rainfall, and regional data

### 🧪 Fertilizer Recommendation
- Suggests appropriate fertilizers based on soil type and nutrient composition

### 🤖 Gen-AI Crop Assistant
- Integrated **Generative AI chatbot** using **Groq (LLaMA-3.1-8B-Instant)**
- Provides disease explanation, remedies, and preventive measures
- Supports **multilingual interaction**:
  - English
  - Hindi
  - Marathi
- ChatGPT-like conversational interface with chat history

---

## 🧠 Technologies Used

### Programming & Frameworks
- Python
- Streamlit

### Machine Learning & Deep Learning
- Convolutional Neural Networks (CNN)
- Random Forest Classifier
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas

### Generative AI
- Groq API
- LLaMA-3.1-8B-Instant / Offline LM studio (meta-llama-3.1-8b-instruct) 
- Prompt Engineering
- Conversational AI

---


---

## ⚙️ How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/cyash21/SmartAgroCare-AI.git
cd SmartAgroCare-AI

```
###2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

###3️⃣ Set Groq API Key
```bash
setx GROQ_API_KEY "your_groq_api_key"
```

###4️⃣ Run the Application
```bash
streamlit run Frontend/app.py
```
