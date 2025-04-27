# 🎬 Movie Rating Predictor

## 📚 Task Objectives
The goal of this project is to **predict movie ratings** using machine learning techniques.  
We use features like movie **Duration**, **Genre**, **Director**, and the **historical success** of the director and genre to build a predictive model.  
The project includes:
- Data preprocessing and feature engineering
- Model training using **Gradient Boosting Regressor**
- Model evaluation through **Cross-Validation** and **Hyperparameter Tuning**
- Model interpretation using **Feature Importance** and **SHAP values**

---

## 🚀 Steps to Run This Project

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/movie-rating-predictor.git
cd movie-rating-predictor
```

2. **Create and Activate a Virtual Environment** (Optional but Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

3. **Install Required Libraries**
```bash
pip install -r requirements.txt
```

4. **Run the Code**
- Make sure the cleaned dataset (`movies_clean.csv`) is placed correctly under `data/processed/`.
- Open and run the main Python file or Jupyter notebook that processes the data, trains the model, and saves outputs (plots, metrics, and model).

Graphs like **feature importance** and **SHAP plots** will automatically be saved in the `outputs/plots/` folder.

---

## 🛠 Clean and Well-Structured Code

- Code is divided into **logical sections**: Preprocessing, Model Training, Evaluation, and Interpretation.
- Follows **PEP8** coding style and good practices.
- Outputs like saved models and graphs are organized into appropriate folders (`models/`, `outputs/`).
- Easy-to-read and maintainable for future improvements or collaboration.
