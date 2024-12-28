import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Data from CSV
# Assuming the CSV file is named 'admission_data.csv' and is in the same directory as the script
csv_file = './data/admission_data.csv'

# Load the data
data = pd.read_csv(csv_file)

# Check the first few rows of the data (optional, for debugging)
print(data.head())

# Step 2: Features and Labels
X = data[['academic_score', 'exam_score', 'extracurricular_score']]
y = data['college']  # Only predict the college, not admission

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest Classifier for College Recommendation (Prediction)
college_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
college_classifier.fit(X_train, y_train)

# Step 5: Save the College Recommendation Model
model_dir = './models'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(college_classifier, os.path.join(model_dir, 'college_model.pkl'))

# Step 6: Evaluate the College Recommendation Model
y_college_pred = college_classifier.predict(X_test)

# Accuracy
college_accuracy = accuracy_score(y_test, y_college_pred)
print(f"College Recommendation Model Accuracy: {college_accuracy:.2f}")

# Detailed Classification Report
print("College Recommendation Model Classification Report:")
print(classification_report(y_test, y_college_pred))

# Confusion Matrix
print("College Recommendation Model Confusion Matrix:")
print(confusion_matrix(y_test, y_college_pred))

print("College model created and saved in the 'models' directory.")
