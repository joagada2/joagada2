# Hi There 👋
I am a data science professional with over 10 years of academic and professional work experience. My specialization and current area of interest is in Machine Learning. I have the skills and experience required to take on machine learning projects from conceptualization, through experimentation, experiment registry/tracking, model serving, and workflow orchestration, to model monitoring/full stack observability, and to deliver machine learning solutions at scale. I have proven work and project experience and a track record of success across various areas of machine learning, from regression, classification, clustering, deep learning, computer vision, and time series forecasting. Other areas of expertise include finetuning and deploying and LLMs, building Retrieval Augmented Generation (RAG), and other Gen AI applications, as well as running inferences with pre-trained models

My current doctoral research focuses on developing domain-specific AI systems that surpass the performance of foundational models in targeted applications. In the materials science domain, I am leveraging comprehensive domain expertise alongside advanced machine learning methodologies to design models that enhance generalization performance, particularly in predicting crystal structures of materials from X-ray diffraction patterns. Additionally, my work in large language models involves advancing their adaptability and precision for specialized tasks by implementing Retrieval-Augmented Generation (RAG) architectures, thereby enabling dynamic updating of domain knowledge. This dual approach not only addresses critical challenges in both fields but also paves the way for more robust, task-oriented AI solutions.

My tech stack includes: Python, R, MATLAB, SQL, Bash, PyTorch, Numpy, Pandas, Searborn, Sckiti-learn, Tensorflow, Keras, Xgboost, FastApi, Flask, Prefect, BERT, NLTK, Transformers, HuggingFace, LangChain, Chroma, Pinecone, Gradio, MLFlow, Dagshub, Git, DVC, Hydra, Heroku, MS Azure, GitHub Action, Streamlit, WhyLabs, SparkML, SparkMLib, AWS, Fbprophet, TorchServe, XLNet, GPT , Docker, Kubernetes, SageMaker, LazyPredict, Power BI, SHAP etc  

# SELECTED PORTFOLIO PROJECT
Below are some selected portfolio projects. There are over 20 other fascinating project repositories

## [Project 1: Machine Learning System for Predicting Fatality in Patients with Myocardial Infarction](https://github.com/joagada2/mi_fatality_prediction)
The first three days of hospitalization for myocardial infarction (MI) patients are critical due to the high risk of fatal complications. Ghafari et al. (2023) developed a machine learning model to predict fatalities during this period. However, the study had two significant issues that I sought to address by building a new model using the same dataset, and then creating a full production-grade machine learning system off the project. First, the study incorrectly treated survivors as the positive class in the training data, which caused sensitivity and specificity to be interchanged, leading to a misreported sensitivity of 0.9435 instead of 0.6923. Second, the paper did not address the common issue of class imbalance in most real-world datasets, failing to use methods like SMOTE, which are proven to enhance model performance. In my model, I correctly designated the positive and negative classes and applied SMOTE to handle class imbalance, thereby significantly improving performance across all metrics to at least 97%. Finally, I developed a full production-grade machine learning system for prediction of outcome in MI patients. The ML system is ready for integration into web or mobile applications. Tools used in this project are as follow:
  - Python (Pandas, matplotlib, seaborn, scikitlearn etc)
  - Lazypredict - for experimentation
  - ExtratreesClassifiers - for model training
  - MLFlow - Experiment registry/tracking
  - FastAPI - Model Serving/API creation
  - AWS EC2 - API deployment
  - AWS ECR - Repository management on AWS
  - GitHub/GitHub Action - Code hosting and CI/CD pipeline
  - WhyLabs - Continous monitoring of model in production
  - Prefect - For workflow orchestration
  - Git - For project version control
  - etc
    
## [Project 2: Machine Learning System for Loan Default Prediction](https://github.com/joagada2/loan-default-prediction-model)
This is a complete end to end machine learning (binary classification) project. It covers all stages of ML lifecycle, from problem selection, model training, experiment tracking, model serving, real time/batch inferencing, model monitoring, workflow orchestration and CI/CD pipeline. Tech/skills used in this project are:
 - Python (Pandas, numpy, matplotlib, seaborn, scikitlearn etc)
 - XGBoost - for model training
 - MLFlow - Experiment tracking
 - dagshub - Experiment tracking, data storage in the cloud and code hosting
 - Hydra - Managing configuration
 - FastAPI - Model Serving
 - Heroku - API exposed on Heroku
 - GitHub/GitHub Action - Code hosting and CI/CD pipeline
 - Streamlit - For creating web app that uses model for prediction
 - WhyLabs - Continous monitoring of model in production
 - Prefect - For workflow orchestration
 - Git - For project version control
 - DVC - For data version control

## [Project 3: Machine Learning System for House Price Prediction](https://github.com/joagada2/king-county-house-price-prediction)
This is a complete end to end machine learning (regression) project. It covers all stages of ML lifecycle, from problem selection, model training, experiment tracking, model serving, real time/batch inferencing, model monitoring, workflow orchestration and CI/CD pipeline. Technologies used include:
 - Python (Pandas, numpy, matplotlib, seaborn, scikitlearn etc)
 - XGBoost - for model training
 - MLFlow - Experiment tracking
 - dagshub - Experiment tracking, data storage in the cloud and code hosting
 - Hydra - Managing configuration
 - FastAPI - Model Serving
 - Heroku - API exposed on Heroku
 - GitHub/GitHub Action - Code hosting and CI/CD pipeline
 - Streamlit - For creating web app that uses model for prediction
 - WhyLabs - Continous monitoring of model in production
 - Prefect - For workflow orchestration
 - Git - For project version control
 - DVC - For data version control
   
## [Project 4: Deep Learning Model for Detection of Malaria Infected Red Blood Cells](https://github.com/joagada2/deep_learning_model_for_detecting_malaria_infected_red_blood_cell)
In this computer vision project, I trained convolutional neural network which takes image of red blood cells and predict if the cell in malaria infected or not. The model has accuracy of 0.94. I also built a flask application for deployment of the model to production environment. Skills/tech used in this project include:
 - Python (pandas, numpy, seaborn, matplotlib)
 - Tensorflow - for model training
 - Keras - Tenforflow API
 - MLFlow - Experiment tracking
 - dagshub - Experiment tracking, data storage in the cloud and code hosting
 - FastAPI - Model Serving
 - Prefect - For workflow orchestration
 - Git - For project version control
 - WhyLabs - Continous monitoring of model in production
 - DVC - For data version control
 - GitHub - Code hosting

## Check my repository for my other fascinating projects


