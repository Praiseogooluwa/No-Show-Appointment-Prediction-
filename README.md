# 📅 No-Show Appointment Prediction  

## 📌 Project Overview  
This project predicts whether a patient will **show up** or **miss** their medical appointment based on historical scheduling data.  
The dataset comes from [Kaggle - Medical Appointment No Shows](https://www.kaggle.com/joniarroba/noshowappointments) and contains patient demographics, scheduling details, and appointment information.  

The main goal is to build a **machine learning model** that can help healthcare providers reduce missed appointments by identifying patients at high risk of not showing up.  

---
### Demo
**Frontend url:** [https://no-show-app.vercel.app](https://no-show-app.vercel.app)

**Backend url:** [https://yourrenderapi.onrender.com](https://yourrenderapi.onrender.com)

---

## 📊 Dataset Description  
The dataset contains the following key columns:  

- **PatientId** – Unique identifier for each patient  
- **AppointmentID** – Unique identifier for each appointment  
- **Gender** – Male or Female  
- **ScheduledDay** – Date and time the appointment was scheduled  
- **AppointmentDay** – Date of the actual appointment  
- **Age** – Age of the patient (in years)  
- **Neighbourhood** – Location of the appointment  
- **Scholarship** – Whether the patient is enrolled in the Bolsa Família program  
- **Hipertension** – Hypertension diagnosis (1 = Yes, 0 = No)  
- **Diabetes** – Diabetes diagnosis (1 = Yes, 0 = No)  
- **Alcoholism** – Alcoholism history (1 = Yes, 0 = No)  
- **Handcap** – Disability level  
- **SMS_received** – Whether the patient received an SMS reminder  
- **No-show** – Target variable (Yes = did not show, No = showed up)  

---

## 🛠 Data Preprocessing  
Steps taken before training:  
1. **Removed duplicates**  
2. **Filtered out invalid ages** (negative values)  
3. **Converted categorical variables** (`Gender`, `No-show`) to numeric values  
4. **Converted date columns** to datetime format  
5. **Created new feature:**  
   - `DaysUntilAppointment` = `AppointmentDay` - `ScheduledDay`  
6. **Encoded** `Neighbourhood` into numeric categories  
7. **Dropped unused ID columns**  

---

## ⚙️ Model Training  
We experimented with:  
- **Random Forest Classifier**  
- **XGBoost Classifier**  

**Imbalance Handling:**  
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.  
- Used **class weights** and `scale_pos_weight` for imbalance-sensitive training.  

**Train-Test Split:**  
- 80% training, 20% testing  
- Stratified sampling to preserve class distribution  

---

## 📈 Model Evaluation  
Metrics used:  
- **Accuracy**  
- **Precision, Recall, F1-score**  
- **Confusion Matrix**  
- **ROC Curve & AUC Score**  

---

### Example output:  

**=== Classification Report ===**

precision recall f1-score support


       0       0.84      0.63      0.72     17715
    
       1       0.25      0.50      0.33      4391


accuracy                           0.61     22106

macro avg 0.54 0.57 0.53 22106

weighted avg 0.72 0.61 0.64 22106

---

### === Confusion Matrix ===

[[11189 6526]

[ 2202 2189]]

**AUC Score:** ~0.58  

---

## 📦 Installation & Usage  

### 1️⃣ Clone the repository  

``bash

git clone https://github.com/Praiseogooluwa/no-show-prediction.git

cd no-show-prediction

### 2️⃣ Install dependencies

pip install -r requirements.txt

### 3️⃣ Train the model

python app_model.py

### 4️⃣ Saved model

**The trained model will be saved as:**
*improved_model.pkl
*scaler.pkl

---
### 📌 Future Improvements

Add external socioeconomic and weather data for better prediction

Experiment with deep learning models

Implement a real-time API for prediction integration

Improve feature engineering (e.g., appointment time slots, day of week)

---

### 👨‍💻 Author
**Isaiah Ogooluwa Bakare – Data Science & Machine Learning Enthusiast**
