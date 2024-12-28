
### Go to the terminal/command prompt of college-prediction

The directory structure will be as follows

├── README.md
├── app
    ├── create_models.py
    └── main.py
├── data
│    └── admission_data.csv
├── models
│    └── college_model.pkl
├── poetry.lock
├── pyproject.toml
└── templates
    └── index.html


### Create virtual env  (IF NOT THERE)
    python3.13 -m venv .venv
### Activate
    source .venv/bin/activate
### Install poetry
    poetry install

### create model (This will create model and save to the models directory)
    python app/create_models.py

### run main program- start the uvicorn server
    python app/main.py

### test

#### web ui 
    http://127.0.0.1:8000 

#### curl command 
curl --location --request POST 'http://127.0.0.1:8000/predict_admission' \
--header 'Content-Type: application/json' \
--data-raw '{
    "academic_score": 10,
    "exam_score": 10,
    "extracurricular_score": 10
}'


### Model tuning concepts (how to make sure our model is accurate?)

#### 1. Accuracy
Definition: The ratio of correctly predicted instances to the total instances.

Formula:

Accuracy = Number of correct Predictions/ Total predictions

Example:
Suppose you have 100 test samples.
The model predicts 90 correctly and 10 incorrectly.  (90/100)=90%


#### 2. Precision
The proportion of true positive predictions out of all positive predictions (how many predicted positives were actually correct).

Precision= True Positives(TP) /(True Positives (TP)+False Positives (FP))

#### 3. Recall (Sensitivity or True Positive Rate)

The proportion of true positives identified out of all actual positives (how many actual positives were detected).

True Positives (TP) / (True Positives (TP)+False Negatives (FN))


### Confusion Matrix
Definition: A table showing the counts of true positives, true negatives, false positives, and false negatives.

Actual/Predicted	Positive	        Negative
Positive	        True Positive (TP)	False Negative (FN)
Negative	        False Positive (FP)	True Negative (TN)


Example:

A model predicts the following for a test set:
50 actual positives: 40 predicted as positive (TP), 10 as negative (FN).
50 actual negatives: 45 predicted as negative (TN), 5 as positive (FP).

Actual/Predicted	Positive	Negative
Positive	        40	           10
Negative	        5	            45

Example Breakdown
Suppose you're predicting whether students pass a test (1 = Pass, 0 = Fail). Your model's results on a test set are:

True Positives (TP): 40 (Predicted Pass, Actually Pass)
False Positives (FP): 5 (Predicted Pass, Actually Fail)
True Negatives (TN): 45 (Predicted Fail, Actually Fail)
False Negatives (FN): 10 (Predicted Fail, Actually Pass)
