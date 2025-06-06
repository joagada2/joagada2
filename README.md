# Hi There 👋
Joseph Oche Agada is a doctoral student at the University of Tennessee working under Dr. Biswas Arpan at the Automation and Steering section of the Independent Research Group 1 (IRG1) at the NSF-funded UT-MRSEC Center for Advanced Material and Manufacturing (CAMM). I am building AI Systems for Autonomous High-throughput discovery and Characterization of Materials with Frustrated Magnets. Materials with frustrated magnets have very high potential in areas of applications like quantum computing due to their rich fundamental physics. However, they are highly sensitive to structural defects; hence, refinement of their structure parameters requires high precision to capture their structural defect, regardless of how subtle they are. I am leveraging various AI algorithms, from LLM/RAG, CNN, Bayesian Optimization, and autoencoder, to build end-to-end domain-specific AI systems from refinement of their structure, from automating the indexing of the structure of such material, identification of defects and defect types, to the structure refinement itself. This work also seeks to automate the discovery and characterization of the materials of interest to meet the demand for analysis of high volumes of data generated from high-throughput diffraction experiments. 

Before joining the University of Tennessee, I was at the forefront of research and development efforts at the Nigerian Air Force. Commissioned into the Service in 2016 as a member of a special research and development team, I was in the team that planned and executed the integration of data and geospatial intelligence into combat operations of the Nigerian Air Force. My team, which I also led at some point, built AI/ML/Data tools for defence and intelligence applications, contributing significantly to the considerable decimation of the notorious Boko Haram and other terror groups in Nigeria at reduced cost. Before joining the Nigerian Air Force, I also taught statistics and data analytics to undergraduates at the University of Nigeria, and participated in mentoring undergraduates.

My tech stack includes: Python, R, MATLAB, SQL, Bash, PyTorch, Numpy, Pandas, Searborn, Sckiti-learn, Tensorflow, Keras, Xgboost, FastApi, Flask, Prefect, BERT, NLTK, Transformers, HuggingFace, LangChain, Chroma, Pinecone, Gradio, MLFlow, Dagshub, Git, DVC, Hydra, Heroku, MS Azure, GitHub Action, Streamlit, WhyLabs, SparkML, SparkMLib, Fbprophet, TorchServe, XLNet, GPT , Docker, Kubernetes, AWS SageMaker, EC2, ECR, S3, DynamoDB, LazyPredict, Power BI, SHAP etcJoseph Oche A

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


