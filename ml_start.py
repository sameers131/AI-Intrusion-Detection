import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as ms
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os


# Load the raw data

files = [
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
]

data = pd.concat([pd.read_csv(file) for file in files])

rows, cols = data.shape
print(f"Number of rows: {rows}")   # 703245
print(f"Number of columns: {cols}")  # 79

col_names = {col: col.strip() for col in data.columns}
data.rename(columns=col_names, inplace=True)


# Clean the data

data.drop_duplicates(inplace=True)

missing_val = data.isna().sum()
print(missing_val.loc[missing_val > 0])

numeric_cols = data.select_dtypes(include=np.number).columns
inf_count = np.isinf(data[numeric_cols]).sum()

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

missing = data.isna().sum()
mis_per = (missing / len(data)) * 100
mis_table = pd.concat([missing, mis_per.round(2)], axis=1)
mis_table = mis_table.rename(columns={0: 'Missing Values', 1: 'Percentage of Total Values'})

med_flow_bytes = data['Flow Bytes/s'].median()
med_flow_packets = data['Flow Packets/s'].median()


# Encode the labels as numbers

label_map = {
    'BENIGN': 0,
    'Bot': 1,
    'PortScan': 2,
    'DDoS': 3
}
data['Label_encoded'] = data['Label'].map(label_map)


# Pick which features to use based on correlation with the label

corr = data.corr(numeric_only=True).round(2)

pos_corr_features = corr['Label_encoded'][
    (corr['Label_encoded'] > 0) & (corr['Label_encoded'] < 1)
].index.tolist()

print("Features with positive correlation with 'Attack Number':\n")
for i, feature in enumerate(pos_corr_features, start=1):
    corr_value = corr.loc[feature, 'Label_encoded']
    print('{:<3} {:<24} :{}'.format(f'{i}.', feature, corr_value))

label_counts = data['Label'].value_counts()
print(label_counts)

LEAKY_OR_ID_COLS = [
    'Label_encoded',
    'Flow ID',
    'Source IP',
    'Destination IP',
    'Timestamp',
]

cols_to_drop = [c for c in LEAKY_OR_ID_COLS if c in data.columns]
print(f"\nDropping leaky/ID columns: {cols_to_drop}")

feature_cols = [f for f in pos_corr_features if f not in cols_to_drop]
print(f"\nUsing {len(feature_cols)} features for modeling:")
print(feature_cols)

X = data[feature_cols].copy()
y = data['Label']


# Scale the features and split into train and test sets

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print("\nTrain class distribution:")
print(y_train.value_counts())
print("\nTest class distribution:")
print(y_test.value_counts())


# Train the Random Forest model

print("\n" + "=" * 60)
print("Training Random Forest...")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print(classification_report(y_test, rf_preds))


# Train the Logistic Regression model

print("\n" + "=" * 60)
print("Training Logistic Regression...")
print("=" * 60)

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
print(classification_report(y_test, lr_preds))


# Compare how well each model catches the rare Bot class

rf_report = classification_report(y_test, rf_preds, output_dict=True)
lr_report = classification_report(y_test, lr_preds, output_dict=True)
print("\nBot-class recall comparison:")
print(f"  Random Forest:       {rf_report['Bot']['recall']:.3f}")
print(f"  Logistic Regression: {lr_report['Bot']['recall']:.3f}")


# Save everything needed for plotting later

os.makedirs('artifacts', exist_ok=True)

joblib.dump(rf_model, 'artifacts/rf_model.pkl')
joblib.dump(lr_model, 'artifacts/lr_model.pkl')
joblib.dump(scaler, 'artifacts/scaler.pkl')
joblib.dump(X_test, 'artifacts/X_test.pkl')
joblib.dump(y_test, 'artifacts/y_test.pkl')
joblib.dump(rf_preds, 'artifacts/rf_preds.pkl')
joblib.dump(lr_preds, 'artifacts/lr_preds.pkl')
joblib.dump(feature_cols, 'artifacts/feature_cols.pkl')
joblib.dump(corr, 'artifacts/corr.pkl')
joblib.dump(label_counts, 'artifacts/label_counts.pkl')

print("\nArtifacts saved to ./artifacts/, run plots.py next.")