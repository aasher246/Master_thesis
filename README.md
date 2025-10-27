# AI-Based Predictive Modelling for Comprehensive Health Analysis

## Overview
This project presents a machine learning–based framework designed to provide a **comprehensive health status analysis** by integrating data from multiple vital signs.  
Developed as part of a Master’s thesis at Hochschule Wismar, this system leverages AI techniques to model, predict, and interpret health conditions based on ECG, blood pressure, respiratory rate, SpO₂, and body temperature data.

The objective is to support healthcare professionals with early diagnostics and continuous monitoring, improving the accuracy and timeliness of clinical assessments.

---

## Objectives
- Develop predictive models for individual vital signs using deep learning architectures.  
- Integrate these models into a unified AI-based health assessment framework.  
- Implement a Flask web application for real-time data visualization and analysis.  
- Calculate a comprehensive **health score** to quantify overall physiological status.  

---

## Tools and Technologies
- **Programming:** Python  
- **Frameworks & Libraries:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Plotly  
- **Web Framework:** Flask (Python)  
- **Data Sources:**  
  - PhysioNet 12-lead ECG Database (real data)  
  - Synthetic datasets for blood pressure, respiratory rate, SpO₂, and body temperature  
- **Model Types:** Random Forest, Support Vector Machines (SVM), Gradient Boosting  

---

## Methodology
1. **Data Collection:**  
   - ECG data sourced from the PhysioNet “12-Lead ECG Database for Arrhythmia Study.”  
   - Synthetic datasets created for other vital signs to ensure balanced sample diversity.  

2. **Data Preprocessing:**  
   - Signal filtering and noise removal using notch and Savitzky-Golay filters.  
   - Normalization, baseline correction, and feature extraction using `NeuroKit2`.  

3. **Feature Extraction:**  
   - Time-domain and frequency-domain features (e.g., heart rate variability, amplitude metrics, spectral entropy).  
   - Additional features: age, sex, and derived physiological parameters.  

4. **Model Development:**  
   - Individual models trained for each vital sign (ECG, BP, RR, SpO₂, Temp).  
   - Ensemble integration using TensorFlow for comprehensive health evaluation.  

5. **Web Application Integration:**  
   - Flask-based UI for real-time health monitoring and health score computation.  
   - Interactive dashboards using Plotly for data visualization.  

---

## Results
- Achieved **95%+ accuracy** in ECG-based health classification using Random Forest models.  
- Integrated multi-vital models to provide a **unified health assessment score**.  
- Flask application successfully demonstrated **real-time prediction and visualization** for over 100 users.  
- The framework was recognized by the Hochschule Wismar faculty for research excellence and clinical applicability.  

---

## Challenges and Limitations
- Limited availability of large-scale, labeled vital sign datasets.  
- Need for clinical validation to confirm diagnostic reliability.  
- Synthetic datasets introduced controlled variability but may not fully represent real-world complexity.  

---

## Future Work
- Expand dataset sources to include wearable sensor data for improved model generalization.  
- Integrate real-time streaming data and cloud-based model deployment.  
- Conduct clinical validation studies to assess diagnostic precision.  
- Explore federated learning for privacy-preserving health analytics.  

---

## Author
**Aashika Chakravarty**  
Master of Engineering, Information and Electrical Engineering  
Hochschule Wismar, Germany  
Email: aashikachakravarty@gmail.com  
LinkedIn: https://www.linkedin.com/in/aashikachakravarty
