⚙️ AQ-MultiCal: 
Multi-Model Machine Learning Platform for Air Quality Sensor Calibration
AQ-MultiCal is a Python-based, web-accessible, and interactive Machine Learning (ML) framework developed to democratize and automate the calibration process of low-cost air quality sensors (LCS). The platform is pollutant-agnostic and enables researchers to evaluate 14 different regression models simultaneously within a unified, no-code environment.

🚀 Live Demo
Access the interactive platform here: https://aq-multical-73g6ufjhbbpplxpvza5efe.streamlit.app/

📂 Sample Dataset: You can access the sample dataset (CSV) required to test the platform in the files section above 

CO2_sensor_data.cvs 

temperature_sensor_data.cvs

humidity_sensor_data.cvs

✨ Key Features
Multi-Model Support: Evaluate 14 ML regression models, including Linear, Tree-based (RF, DT), and Boosting (XGBoost, LightGBM, CatBoost) approaches.
Automated Optimization: Integrated hyperparameter tuning using Grid Search, Randomized Search, and Bayesian Optimization.
Interactive Visualization: Dynamic performance analytics including scatter plots, time-series analysis, residual distributions (KDE/Histogram), and parallel coordinates.
No-Code Workflow: End-to-end processing from CSV data ingestion to model evaluation without requiring programming expertise.
Resource Awareness: Includes computational cost analysis (processing time) to evaluate model feasibility for real-time applications.

📊 Methodology & WorkflowThe platform architecture manages the entire data lifecycle:
Data Ingestion: Uploading synchronized pollutant and environmental (Temp/Hum) datasets.Preprocessing: Feature scaling (StandardScaler) and versatile data splitting (Time-Based or Random).Optimization: Systematic hyperparameter search to maximize $R^2$ and minimize $RMSE$.Validation: Robust performance estimation using 5-fold cross-validation.

🛠️ Installation & Local Usage
To run the platform on your local machine:

Clone the repository: https://github.com/tastan45/AQ-MultiCal.git

Install dependencies: pip install -r requirements.txt

Run the application: streamlit run AQ-MultiCal.py

👨‍🏫 Author
Assoc. Prof. Dr. Mehmet TAŞTAN

E-mail: mehmet.tastan@cbu.edu.tr

Manisa Celal Bayar University

Department of Electronics and Automation Manisa/TURKEY

Specializing in IoT, Machine Learning, and Environmental Monitoring.
