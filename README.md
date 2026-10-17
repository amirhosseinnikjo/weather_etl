\# 🌦️ Weather ETL \& Machine Learning



\*\*From Raw Weather Data to Machine Learning Insights\*\*



An end-to-end Python project that collects weather data, processes it through an ETL pipeline, stores it in DuckDB, and applies machine learning techniques for weather prediction and classification.



The project combines \*\*Data Engineering, Machine Learning, and Data Visualization\*\* in a practical workflow built around real-world weather data.



\---



\## 📌 Project Overview



This project started as a weather data pipeline and evolved into a broader data analytics and machine learning project.



It demonstrates how raw data can be collected from an external API, transformed into a structured dataset, stored in an analytical database, and used to explore predictive models and interactive visualizations.



\### Key Features



\- 🌐 \*\*Weather Data Extraction:\*\* Retrieve weather data from the Open-Meteo API.

\- ⚙️ \*\*ETL Pipeline:\*\* Extract, transform, and load data using Python.

\- 🗄️ \*\*Data Storage:\*\* Store and query weather records using DuckDB.

\- 🤖 \*\*Weather Prediction:\*\* Experiment with Random Forest regression to predict weather variables.

\- 🏷️ \*\*Weather Classification:\*\* Explore machine learning classification for weather conditions.

\- 📊 \*\*Interactive Dashboard:\*\* Visualize weather data and analysis through Streamlit.

\- ⏱️ \*\*Automation:\*\* Schedule recurring ETL runs using Python.



\---



\## 🏗️ ETL Pipeline



The pipeline follows the standard Extract, Transform, and Load architecture.



\*\*Extract → Transform → Load\*\*



\### 1. Extract



Retrieve weather data from the Open-Meteo API using Python and Requests.



\### 2. Transform



Process the collected data with Python and Pandas, preparing it for storage and analysis.



\### 3. Load



Store the processed weather records in DuckDB, an analytical database that supports efficient local querying.



\---



\## 🤖 Machine Learning



The project extends beyond data collection by applying machine learning to weather data.



\### Weather Prediction — Random Forest Regression



A Random Forest regression model is used to explore the prediction of numerical weather variables from historical observations and engineered features.



The workflow includes:



\- Preparing features and target variables.

\- Creating time-based features and lag features.

\- Splitting data into training and testing sets chronologically.

\- Training and evaluating the regression model.

\- Measuring performance with metrics such as MAE and R².



\### Weather Classification



A classification model is also included to explore categorizing weather conditions based on available data.



This component provides a foundation for investigating how historical weather observations can be used to identify weather patterns.



\*Model performance depends on the quality of the data, selected features, and evaluation methodology.\*



\---



\## 📊 Data Visualization



The project includes a Streamlit dashboard for exploring weather data through an interactive interface.



The dashboard provides a starting point for presenting collected data and analytical results in a more accessible format.



\---



\## 🛠️ Technology Stack



| Technology | Purpose |

|---|---|

| Python | Core programming language |

| Requests | API communication and data extraction |

| Pandas | Data processing and feature engineering |

| DuckDB | Local analytical database |

| Scikit-learn | Machine learning and model evaluation |

| Streamlit | Interactive dashboard |

| Open-Meteo API | Weather data source |



\---



\## 📁 Project Structure



```text

weather\_etl/

├── database/

│   └── Weather.duckdb

├── script/

│   ├── extract.py

│   ├── transform.py

│   ├── load.py

│   ├── main.py

│   ├── Schedule.py

│   ├── histroical.py

│   ├── random\_forest.py

│   ├── classifier.py

│   └── app.py

├── .gitignore

└── README.md

```



\---



\## 🚀 Getting Started



\### Prerequisites



\- Python 3.10 or newer

\- Git

\- Internet access for retrieving weather data



\### 1. Clone the Repository



```bash

git clone https://github.com/amirhosseinnikjo/weather\_etl.git

cd weather\_etl

```



\### 2. Create a Virtual Environment



```bash

python -m venv .venv

```



Activate it on Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\### 3. Install Dependencies



```bash

pip install requests pandas duckdb scikit-learn streamlit

```



\### 4. Run the ETL Pipeline



```bash

python script/main.py

```



\### 5. Run the Dashboard



```bash

streamlit run script/app.py

```



> \*\*Note:\*\* The dashboard and machine learning scripts may require the appropriate database and input data to be available. Verify the file paths in the scripts before running them.



\---



\## 🎯 Project Goals



\- Build a practical ETL workflow using a real-world API.

\- Practice data transformation and analytical storage.

\- Explore feature engineering and supervised machine learning.

\- Evaluate predictive models using appropriate metrics.

\- Present data through an interactive dashboard.

\- Develop a foundation for more advanced data engineering and intelligent data systems.



\---



\## 🔮 Future Improvements



\- Add automated data quality checks and validation.

\- Improve feature engineering and model evaluation.

\- Add structured logging and robust error handling.

\- Improve dashboard design and analytical visualizations.

\- Add automated tests and dependency management.

\- Document model results, limitations, and reproducible experiments.



\---



\## 👨‍💻 Author



\*\*Amirhossein Nikjoo\*\*



Computer Engineering Student | Data Engineering \& Machine Learning



\[GitHub Profile](https://github.com/amirhosseinnikjo)



\---



\*Built as a hands-on project to explore the journey from raw data pipelines to intelligent systems.\*

