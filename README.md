# Prognosis
> **Foreknowledge is Power.**

Prognosis is a high-performance Machine Learning application designed to predict the risk of diabetes based on key metabolic health indicators. It uses a separated architecture with a blazing-fast Python FastAPI backend and a clean, responsive, vanilla web interface.

## 🚀 Features
* **AI-Powered Predictions:** Utilizes a trained Random Forest Classifier to assess diabetes risk.
* **Advanced Feature Engineering:** Incorporates interaction terms (BMI × Age, Glucose × BMI) for higher model accuracy.
* **Calibrated Risk Threshold:** Uses a specific probability threshold (0.2424) for sensitive and highly accurate risk categorization.
* **Modern UI:** A fully responsive frontend featuring conditional color-coding (red/green) and dynamic risk-meter animations.
* **Separated Architecture:** Clean separation of concerns between the frontend (HTML/JS) and backend (FastAPI).

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Scikit-Learn, Pandas, Uvicorn
* **Frontend:** HTML5, CSS3, Vanilla JavaScript 
* **Typography:** Outfit (Headings) & Plus Jakarta Sans (Body)
* **Modeling:** Jupyter Notebook (`main.ipynb`) for data exploration and model training.

## 📂 Project Structure
```text
PROJECT/
│
├── .vscode/                 # VS Code workspace settings
│
├── BackEnd/                 # FastAPI Backend & ML Assets
│   ├── __pycache__/
│   ├── app.py               # Main FastAPI application
│   ├── diabetes_model.pkl   # Trained Random Forest model
│   ├── requirements         # Python dependencies 
│   └── scaler.pkl           # StandardScaler for data normalization
│
├── Interface/               # Frontend Client
│   └── interface.html       # Main UI 
│
├── .gitignore               # Git ignore rules
├── diabetes.csv             # Training dataset
└── main.ipynb               # Model training and evaluation notebook
```

## 🚀 Getting Started & Deployment

To run Prognosis locally, you must start the backend server and open the frontend interface.

**1. Launch the Backend API**
    Open your terminal, navigate to the backend directory, install the dependencies, and start the server:

  1. Start the Backend (API)
     The frontend requires the backend API to be running to process predictions.
     Open your terminal and navigate to the BackEnd directory:
       cd BackEnd

   2. Install the required dependencies:
      pip install -r requirements

      (Note: Ensure your requirements file contains fastapi, uvicorn, scikit-learn, and pandas)

  3. Start the FastAPI server:
      uvicorn app:app --reload ( In terminal )

  4. The API will now be running on http://127.0.0.1:8000. Leave this terminal open.

**2. Start the Frontend (UI)**
      Because the frontend is built with vanilla web technologies, no build steps are required.
  
  Navigate to the Interface/ folder.

  Open interface.html directly in any modern web browser by double-clicking it, or use Live Server in VS Code for hot-reloading.

  Enter your health metrics and click "Calculate Risk Score" to see the model in action!

☁️ Deployment
(Coming Soon)
The backend will be deployed to Render as a web service, and the frontend will be hosted securely for public access.
