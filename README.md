# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*:URMIL RAMESHRAO BHOYAR 

*INTERN ID*:CT06DL1258

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*:6 WEEKS

*MENTOR*: NEELA SANTOSH

# DESCRIPTION OF TASK 

# Spam Email Detection using Scikit-learn
**Project Overview**
This project, CREATE A PREDICTIVE MODEL USING SCIKIT-LEARN TO CLASSIFY OR PREDICT OUTCOMES FROM A DATASET (E.G., SPAM EMAIL DETECTION), demonstrates how to build a machine learning model using Scikit-learn to 
identify whether an incoming email is spam or ham (not spam).
By leveraging real-world labeled data and using text classification techniques, this model converts raw email messages into numerical features and predicts the likelihood of an email being spam. This helps in 
automating email filtering systems, improving user experience, and enhancing cybersecurity.
The entire project is implemented in Jupyter Notebook, which provides an interactive coding environment perfect for data preprocessing, training, evaluation, and testing machine learning models.

**Tools & Technologies Used**
1. Programming Language
Python 3: Used for all data manipulation, training, and evaluation tasks.

2. Libraries
Pandas: To read and clean data.
NumPy: For efficient numerical computation.
Scikit-learn: For ML model building (TfidfVectorizer, LogisticRegression, train-test split, accuracy_score).
TfidfVectorizer: Used for converting text data to feature vectors.

3. Code Editor
Jupyter Notebook: A powerful, browser-based Python editor ideal for ML and data science experiments.

**Workflow**
Step 1: Loading and Preparing the Data
The CSV file mail_data.csv is loaded using pandas.read_csv().
Null values are replaced with empty strings to handle missing entries.

Step 2: Label Encoding
Email labels are encoded as:
spam → 0
ham → 1

Step 3: Splitting the Dataset
Data is split into features (X) and labels (Y).
We use an 80/20 train-test split to prepare the model for evaluation.

Step 4: Feature Extraction
TfidfVectorizer is used to convert text data into numerical vectors.
Common stopwords are removed and text is lowercased.

Step 5: Model Training
Logistic Regression is used as the classification algorithm.
The model is trained using the training feature vectors and labels.

Step 6: Model Evaluation
Accuracy is calculated on both the training and test datasets using accuracy_score.

Step 7: Real-Time Prediction
A system is built to take user input (email message), transform it using the same vectorizer, and predict whether it’s spam or ham.

**Learning Objectives**
By completing this project, users will learn:

1) How to clean and prepare textual data for machine learning.
2) Using TfidfVectorizer for NLP-based feature extraction.
3) Splitting data into training and testing sets effectively.
4) Training a classification model using logistic regression.
5) Evaluating a model using performance metrics like accuracy.
6) Implementing real-time predictions for user input.

**Use Cases and Applications**
1) Email Filtering Systems: Automatically detect and filter spam messages.
2) Customer Support Bots: Distinguish spam inputs in live chat.
3) Security Tools: Detect phishing or unsolicited messages in an enterprise environment.
4) Text Classification Projects: Serve as a base for broader NLP applications like sentiment analysis.

**Future Enhancements**
1) Add a web interface using Flask or Streamlit.
2) Train with a larger and more balanced dataset.
3) Compare with other models like Naive Bayes, SVM, or Random Forest.
4) Apply additional NLP preprocessing such as lemmatization or n-grams.
5) Export the model and use it in production environments (e.g., integrated into Gmail filters).

**Code Execution**

Since this project is implemented in Jupyter Notebook, follow the steps below to execute:

1) Open Jupyter Notebook (locally or through Anaconda Navigator).
2) Open the .ipynb file containing the code.
3) Make sure to have the required libraries installed:
bash
Copy
Edit
pip install numpy pandas scikit-learn
4) Replace the CSV file path (mail_data.csv) with the correct local path.
5) Run each cell step-by-step to see outputs.
6) For real-time prediction, enter any email content when prompted.

# output 
![Image](https://github.com/user-attachments/assets/52782238-7e9e-45c6-a7dd-906e165c9052)

![Image](https://github.com/user-attachments/assets/1a3a3ee3-c791-4a3b-978d-d2aab1696e5c)

![Image](https://github.com/user-attachments/assets/911b9cfd-adb5-4a06-9e42-29d0a50e0e91)

![Image](https://github.com/user-attachments/assets/a8267ceb-e62e-4699-af59-dff6313b090b)

![Image](https://github.com/user-attachments/assets/60a189d8-7b33-4d97-9f50-8cf3f81ad157)

![Image](https://github.com/user-attachments/assets/a64c9978-86f7-40cb-ac1c-201479c7fb3e)

![Image](https://github.com/user-attachments/assets/592ae821-9757-47f2-8f61-e4a5202fc217)

![Image](https://github.com/user-attachments/assets/5bdbfbf8-9d0f-417c-bee7-0a261a4498aa)

