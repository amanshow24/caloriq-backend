# 🔥 CaloriQ – Calorie Burn Predictor

CaloriQ is a full-stack web application that predicts calories burned based on user input such as age, gender, height, weight, body temperature, and heart rate. It features a machine learning model (scikit-learn) on the backend and a sleek React interface on the frontend.

---

## 🚀 Live Demo 

 🔗 [Live Demo]   https://caloriq-ten.vercel.app/

---

## ✨ Features

- 🔮 Predicts calories burned using a trained ML model
- 🧠 Flask backend with scikit-learn model inference
- ⚛️ React frontend with smooth routing and user input forms
- 📊 Based on real calorie dataset (`calories.csv`)
- 🎯 Clean UI and responsive design
- 🧪 Model training script included (`train_model.py`)

---

## 🛠️ Tech Stack

**Frontend:**
- React.js (Vite)
- JavaScript
- CSS (custom and responsive)

**Backend:**
- Python
- Flask
- scikit-learn
- Pandas
- Pickle (for model storage)

**Deployment:**
- Backend: Render.com (`render.yaml` included)
- Frontend: Vercel / Netlify / GitHub Pages (Vite-friendly)

---

## 📦 Installation

### 🖥️ Frontend Setup

```bash
git clone https://github.com/yourusername/caloriq-frontend.git
cd caloriq-frontend-main
npm install
npm run dev
````

Frontend will be running on `http://localhost:5173` (default Vite port)

---

### 🧠 Backend Setup

```bash
git clone https://github.com/yourusername/caloriq-backend.git
cd caloriq-backend-main
pip install -r requirements.txt
python app.py
```

Backend will be running on `http://localhost:5000`

Make sure both ports are open and CORS is handled if testing locally.

---

## ▶️ Usage

1. Launch both backend (`app.py`) and frontend (`npm run dev`)
2. Open the frontend in your browser
3. Enter your details: gender, age, height, weight, body temperature, and heart rate
4. Get your predicted calorie burn instantly

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the project
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -m 'Added feature'`
4. Push to GitHub: `git push origin feature/new-feature`
5. Open a pull request

Please follow the existing code style and structure.

---

## 📄 License

Licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

* [scikit-learn](https://scikit-learn.org/)
* [Pandas](https://pandas.pydata.org/)
* [Flask](https://flask.palletsprojects.com/)
* [React](https://reactjs.org/)
* [Vite](https://vitejs.dev/)
* Dataset used: `calories.csv`

---

> Built with ❤️ by Aman Show
