🥔 Potato Leaf Disease Classification 🌱

<img width="1272" height="627" alt="image" src="https://github.com/user-attachments/assets/29e37af4-32f6-466d-a446-4aa9313c8470" />



This project is a Potato Leaf Disease Classification System that detects Early Blight and Late Blight from potato leaf images using Deep Learning.
It features a FastAPI backend for model inference and a React (Material-UI) frontend for an interactive user experience.

🚀 Features

Upload a potato leaf image 🌿

Get real-time predictions with confidence scores 📊

Detects Early Blight and Late Blight

Built with TensorFlow/Keras, FastAPI, and React (Material-UI)

🛠️ Tech Stack

Frontend: React, Material-UI

Backend: FastAPI, Python

Machine Learning: TensorFlow/Keras, CNN Model

Deployment: (optional → add Docker, Render, Vercel if you plan later)

📂 Project Structure
project/
│── backend/         # FastAPI backend with ML model
│── frontend/        # React frontend (Material-UI)
│── saved_models/    # Trained models (.keras)
│── README.md        # Documentation



⚙️ Installation & Setup
1. Clone the repository

git clone https://github.com/your-username/potato-disease-classification.git
cd potato-disease-classification



2. Backend (FastAPI)
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload

3. Frontend (React)
   cd frontend
   npm install  
   npm start

📸 Usage

Open the web app in your browser.

Upload a potato leaf image.

Get instant prediction (Early Blight / Late Blight) with confidence percentage.

📊 Dataset

The model is trained on the PlantVillage dataset (Potato Leaves).

🙌 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss.

📜 License

This project is licensed under the MIT License.

