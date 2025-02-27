import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder ,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)


# Step 1: Read the data
def read_data(file_name):
    file_path = f"{parent_dir}/data/{file_name}"
    print(f"Trying to read: {file_path}")  
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)

    else:
        return None


# Step 2: Preprocess the data

def preprocess_data(df, target_column, scaler_type):
    """
    Prepares the dataset for training:
    - Converts categorical features to numerical using Label Encoding
    - Scales numerical features
    - Splits the dataset into train and test sets
    """

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Apply Label Encoding to categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders if needed later

    # Split into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Apply scaling
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_scaled = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Step 3: Train the model
def train_model(X_train, y_train, model, model_name):
    model.fit(X_train, y_train)

    # Ensure the directory for trained models exists
    model_dir = os.path.join(parent_dir, "trained_model")
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    return model


# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return round(accuracy_score(y_test, y_pred), 2)
