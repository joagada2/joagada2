# Hi There 👋
Joseph Oche Agada is a doctoral student at the University of Tennessee, working under the supervision of Dr. Arpan Biswas in the Automation and Steering division of Independent Research Group 1 (IRG1) at the NSF-funded Center for Advanced Materials and Manufacturing (CAMM), UT-MRSEC. His research focuses on the development of AI-driven systems for the autonomous, high-throughput discovery and characterization of materials with frustrated magnetism. These materials, known for their rich emergent physics, hold significant promise for applications in quantum computing and spintronic devices. However, their physical properties are exceptionally sensitive to subtle structural defects, necessitating ultra-precise refinement of structural parameters.

Joseph's work integrates a suite of artificial intelligence approaches—including Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG), Convolutional Neural Networks (CNNs), Bayesian Optimization, and Autoencoders—to build domain-specific, end-to-end AI systems for structural analysis. His pipeline encompasses automated crystallographic indexing, defect detection and classification, and full structural refinement. The goal is to enable autonomous workflows capable of handling the high data volumes generated from modern high-throughput diffraction experiments, thereby accelerating the discovery cycle for quantum and topological materials.

Prior to his Ph.D. studies, Joseph served in the Nigerian Air Force, where he was a core member—and later leader—of a specialized research and development team focused on integrating AI, machine learning, and geospatial intelligence into military operations. Commissioned in 2016, his team played a critical role in designing and deploying data-centric tools that significantly enhanced operational efficiency and contributed to the strategic dismantling of Boko Haram and related insurgent groups in Nigeria. He also previously lectured in statistics and data analytics at the University of Nigeria, where he actively mentored undergraduate students.


Tech Stack: Python, R, MATLAB, SQL, Git, Bash, PyTorch, Numpy, Pandas, Searborn, Sckiti-learn, Tensorflow, Keras, Xgboost, FastApi, Flask, Prefect, BERT, NLTK, Transformers, HuggingFace, LangChain, Chroma, Pinecone, gpt, claude, gemini, Llama, Gradio, MLFlow, Dagshub, DVC, Hydra, Heroku, MS Azure, GitHub Action, Streamlit, WhyLabs, fbprophet , Docker, Kubernetes, AWS SageMaker, AWS EC2, AWS ECR, AWS S3, AWS DynamoDB, LazyPredict, Power BI, SHAP, Redis, React.js, BeautifulSoup, joblib, PowerBI, Scipy, Nginx etc

# SELECTED PORTFOLIO PROJECT
Below are some selected portfolio projects. There are over 20 other fascinating project repositories

## [Project 1: Conversational RAG Agent for Web Content Interaction and Navigation](https://github.com/joagada2/dse_697_fianl_project)
The University of Tennessee, Knoxville (UTK) maintains a vast digital footprint with over 19,000 pages spread across its main and sub-domains (utk.edu). Thousands of visitors around the globe scan through the thousands of pages on utk.edu, reading chunks of irrelevant text to get the pieces of information they need. Visitors to utk.edu are also often interested in content of UT Systems website (tennessee.edu) and Oak Ridge National Laboratory website (ornl.gov). In this project, we built a chatbot that provides natural language interface for users to chat with content on utk.edu, tennessee.edu and ornl.gov. The system provides chat response in conversational manner alongside links to pages where information were extracted from on the websites. It is hope to make access to information on those websites more conversational and seamless.
## APPLICATION LINKS
 - Frontend [here](http://3.144.96.138/)
 - Backend [here](http://3.143.23.19:8000/docs)

## [Project 2: Machine Learning System for Predicting Fatality in Patients with Myocardial Infarction](https://github.com/joagada2/mi_fatality_prediction)
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
    
## [Project 3: Machine Learning System for Loan Default Prediction](https://github.com/joagada2/loan-default-prediction-model)
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

## [Project 4: Machine Learning System for House Price Prediction](https://github.com/joagada2/king-county-house-price-prediction)
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
   
## [Project 5: Deep Learning Model for Detection of Malaria Infected Red Blood Cells](https://github.com/joagada2/deep_learning_model_for_detecting_malaria_infected_red_blood_cell)
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


