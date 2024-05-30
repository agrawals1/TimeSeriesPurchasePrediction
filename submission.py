# import pandas as pd
# import numpy as np
# from datetime import datetime
# # from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# # Load data
# data = pd.read_csv('/home/shubham/2021_6sense_DS_Takehome_Challenge/training.tsv', delimiter='\t', header=None, names=['user_id', 'date', 'activity'])
# data_test = pd.read_csv('/home/shubham/2021_6sense_DS_Takehome_Challenge/test.tsv', delimiter='\t', header=None, names=['user_id', 'date', 'activity'])
# # unique_users = data['user_id'].unique()

# # # Sample 20% of the users
# # sampled_users = np.random.choice(unique_users, size=int(len(unique_users) * 0.2), replace=False)

# # # Filter data to retain only the sampled users
# # data = data[data['user_id'].isin(sampled_users)]

# import re
# # Convert date column to datetime
# data['date'] = pd.to_datetime(data['date'], errors='coerce')
# data_test['date'] = pd.to_datetime(data_test['date'], errors='coerce')

# # Convert activity column to lowercase
# data['activity'] = data['activity'].str.lower()
# data_test['activity'] = data_test['activity'].str.lower()

# # 1. Check for null values
# null_values = data.isnull().sum()
# null_values_test = data_test.isnull().sum()
# print("Null Values:\n", null_values)
# print("Null Values Test:\n", null_values_test)

# # 2. Validate date column
# invalid_dates = data[data['date'].isnull()]
# invalid_dates_test = data_test[data_test['date'].isnull()]
# print("Invalid Dates:\n", invalid_dates)
# print("Invalid Dates Test:\n", invalid_dates_test)

# # 3. Validate activity column
# non_string_activities = data[~data['activity'].apply(lambda x: isinstance(x, str))]
# non_string_activities_test = data_test[~data_test['activity'].apply(lambda x: isinstance(x, str))]
# print("Non-String Activities:\n", non_string_activities_test)

# # 4. Validate user_id column
# # Check for alphanumeric values and consistent length
# user_id_pattern = re.compile(r'^[a-zA-Z0-9]+$')
# valid_user_ids = data['user_id'].apply(lambda x: bool(user_id_pattern.match(x)))
# consistent_length = data['user_id'].apply(lambda x: len(x)).nunique() == 1
# valid_user_ids_test = data_test['user_id'].apply(lambda x: bool(user_id_pattern.match(x)))
# consistent_length_test = data_test['user_id'].apply(lambda x: len(x)).nunique() == 1

# print("Are all user IDs alphanumeric?", valid_user_ids.all())
# print("Are all user IDs of consistent length?", consistent_length)
# print("Are all user IDs alphanumeric Test?", valid_user_ids_test.all())
# print("Are all user IDs of consistent length Test?", consistent_length_test)

# # For detailed inspection of invalid user IDs (if any)
# invalid_user_ids = data[~valid_user_ids]
# print("Invalid User IDs:\n", invalid_user_ids)
# invalid_user_ids_test = data_test[~valid_user_ids_test]
# print("Invalid User IDs Test:\n", invalid_user_ids_test)
# print(data['activity'].unique())
# print(data_test['activity'].unique())

# import numpy as np
# # Create target variable: Purchase indicator
# data['purchase'] = np.where(data['activity'] == 'purchase', 1, 0)
# # data_test['purchase'] = np.where(data_test['activity'] == 'purchase', 1, 0)

# # Aggregate data by user_id
# user_agg = data.groupby('user_id').agg({
#     'activity': lambda x: x.tolist(),
#     'date': lambda x: x.tolist(),
#     'purchase': 'sum'
# }).reset_index()

# # Aggregate data by user_id
# user_agg_test = data_test.groupby('user_id').agg({
#     'activity': lambda x: x.tolist(),
#     'date': lambda x: x.tolist()
# }).reset_index()

# # Feature Engineering: Activity counts, recent activity, etc.
# user_agg['email_open_count'] = user_agg['activity'].apply(lambda x: x.count('emailopen'))
# user_agg['form_submit_count'] = user_agg['activity'].apply(lambda x: x.count('formsubmit'))
# user_agg['email_clickthrough_count'] = user_agg['activity'].apply(lambda x: x.count('emailclickthrough'))
# user_agg['customer_support_count'] = user_agg['activity'].apply(lambda x: x.count('customersupport'))
# user_agg['page_view_count'] = user_agg['activity'].apply(lambda x: x.count('pageview'))
# user_agg['web_visit_count'] = user_agg['activity'].apply(lambda x: x.count('webvisit'))
# user_agg['activity_duration'] = user_agg.apply(
#     lambda x: ((max(x['date']) - min(x['date'])).days),
#     axis=1
# )

# user_agg_test['email_open_count'] = user_agg_test['activity'].apply(lambda x: x.count('emailopen'))
# user_agg_test['form_submit_count'] = user_agg_test['activity'].apply(lambda x: x.count('formsubmit'))
# user_agg_test['email_clickthrough_count'] = user_agg_test['activity'].apply(lambda x: x.count('emailclickthrough'))
# user_agg_test['customer_support_count'] = user_agg_test['activity'].apply(lambda x: x.count('customersupport'))
# user_agg_test['page_view_count'] = user_agg_test['activity'].apply(lambda x: x.count('pageview'))
# user_agg_test['web_visit_count'] = user_agg_test['activity'].apply(lambda x: x.count('webvisit'))
# user_agg_test['activity_duration'] = user_agg_test.apply(
#     lambda x: ((max(x['date']) - min(x['date'])).days),
#     axis=1
# )

# # Target variable
# user_agg['purchase_target'] = user_agg['purchase'].apply(lambda x: 1 if x > 0 else 0)
# # user_agg_test['purchase_target'] = user_agg_test['purchase'].apply(lambda x: 1 if x > 0 else 0)

# # Drop unnecessary columns
# user_agg.drop(columns=['activity', 'date'], inplace=True)
# user_agg_test.drop(columns=['activity', 'date'], inplace=True)

# # # Train-Test Split based on user_ids
# # user_ids = user_agg['user_id'].unique()
# # train_user_ids, test_user_ids = train_test_split(user_ids, test_size=0.2, random_state=42)

# # train_data = user_agg[user_agg['user_id'].isin(train_user_ids)]
# # test_data = user_agg[user_agg['user_id'].isin(test_user_ids)]
# train_data = user_agg
# test_data = user_agg_test

# X_train = train_data.drop(columns=['purchase_target', 'user_id', 'purchase'])
# y_train = train_data['purchase_target']
# X_test = test_data.drop(columns=['user_id'])
# # y_test = test_data['purchase_target']

# # Sample weights based on purchase quantity
# weights_train = train_data['purchase'] + 1  # Adding 1 to avoid zero weights
# # weights_test = test_data['purchase'] + 1

# # Model Training with Sample Weights
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train, sample_weight=weights_train)

# # Predictions
# y_pred = model.predict(X_test)
# # y_pred_proba = model.predict_proba(X_test)[:, 1] 

# # # Evaluation
# # print(classification_report(y_test, y_pred))
# # print(confusion_matrix(y_test, y_pred))
# # print('ROC-AUC Score:', roc_auc_score(y_test, y_pred_proba))
# ------------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# from sklearn.preprocessing import StandardScaler

# # Load data
# data = pd.read_csv('/home/shubham/2021_6sense_DS_Takehome_Challenge/training.tsv', delimiter='\t', header=None, names=['user_id', 'date', 'activity'])
# data_test = pd.read_csv('/home/shubham/2021_6sense_DS_Takehome_Challenge/test.tsv', delimiter='\t', header=None, names=['user_id', 'date', 'activity'])

# import re
# # Convert date column to datetime
# data['date'] = pd.to_datetime(data['date'], errors='coerce')
# data_test['date'] = pd.to_datetime(data_test['date'], errors='coerce')

# # Convert activity column to lowercase
# data['activity'] = data['activity'].str.lower()
# data_test['activity'] = data_test['activity'].str.lower()

# # Create target variable: Purchase indicator
# data['purchase'] = np.where(data['activity'] == 'purchase', 1, 0)

# # Aggregate data by user_id
# user_agg = data.groupby('user_id').agg({
#     'activity': lambda x: x.tolist(),
#     'date': lambda x: x.tolist(),
#     'purchase': 'sum'
# }).reset_index()

# user_agg_test = data_test.groupby('user_id').agg({
#     'activity': lambda x: x.tolist(),
#     'date': lambda x: x.tolist()
# }).reset_index()

# # Feature Engineering: Activity counts, recent activity, etc.
# user_agg['email_open_count'] = user_agg['activity'].apply(lambda x: x.count('emailopen'))
# user_agg['form_submit_count'] = user_agg['activity'].apply(lambda x: x.count('formsubmit'))
# user_agg['email_clickthrough_count'] = user_agg['activity'].apply(lambda x: x.count('emailclickthrough'))
# user_agg['customer_support_count'] = user_agg['activity'].apply(lambda x: x.count('customersupport'))
# user_agg['page_view_count'] = user_agg['activity'].apply(lambda x: x.count('pageview'))
# user_agg['web_visit_count'] = user_agg['activity'].apply(lambda x: x.count('webvisit'))
# user_agg['activity_duration'] = user_agg.apply(
#     lambda x: ((max(x['date']) - min(x['date'])).days),
#     axis=1
# )

# user_agg_test['email_open_count'] = user_agg_test['activity'].apply(lambda x: x.count('emailopen'))
# user_agg_test['form_submit_count'] = user_agg_test['activity'].apply(lambda x: x.count('formsubmit'))
# user_agg_test['email_clickthrough_count'] = user_agg_test['activity'].apply(lambda x: x.count('emailclickthrough'))
# user_agg_test['customer_support_count'] = user_agg_test['activity'].apply(lambda x: x.count('customersupport'))
# user_agg_test['page_view_count'] = user_agg_test['activity'].apply(lambda x: x.count('pageview'))
# user_agg_test['web_visit_count'] = user_agg_test['activity'].apply(lambda x: x.count('webvisit'))
# user_agg_test['activity_duration'] = user_agg_test.apply(
#     lambda x: ((max(x['date']) - min(x['date'])).days),
#     axis=1
# )

# # Target variable
# user_agg['purchase_target'] = user_agg['purchase'].apply(lambda x: 1 if x > 0 else 0)

# # Drop unnecessary columns
# user_agg.drop(columns=['activity', 'date'], inplace=True)
# user_agg_test.drop(columns=['activity', 'date'], inplace=True)

# # Train data
# train_data = user_agg
# test_data = user_agg_test

# # Separate features and target
# X_train = train_data.drop(columns=['purchase_target', 'user_id', 'purchase'])
# y_train = train_data['purchase_target']
# X_test = test_data.drop(columns=['user_id'])

# # Normalize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Handle class imbalance by undersampling the majority class
# purchase_indices = np.where(y_train == 1)[0]
# non_purchase_indices = np.where(y_train == 0)[0]
# random_non_purchase_indices = np.random.choice(non_purchase_indices, size=len(purchase_indices), replace=False)
# balanced_indices = np.concatenate([purchase_indices, random_non_purchase_indices])

# X_train_balanced = X_train[balanced_indices]
# y_train_balanced = y_train.iloc[balanced_indices]

# # Calculate feature weights based on the number of purchases
# feature_weights = train_data['purchase'].values + 1  # Adding 1 to avoid zero weights

# # Normalize the feature weights to ensure they are in a reasonable range
# feature_weights = feature_weights / feature_weights.max()

# # Model Training with weighted features
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train_balanced, y_train_balanced, sample_weight=feature_weights[balanced_indices])


# # Predictions
# y_pred = model.predict(X_test)
# # y_pred_proba = model.predict_proba(X_test)[:, 1]

# # # Evaluation on the training set (for demonstration)
# # y_train_pred = model.predict(X_train_balanced)
# # print(classification_report(y_train_balanced, y_train_pred))
# # print(confusion_matrix(y_train_balanced, y_train_pred))
# # print('ROC-AUC Score:', roc_auc_score(y_train_balanced, model.predict_proba(X_train_balanced)[:, 1]))

# # For the test set, assuming the true labels are not available
# print("Predictions on test set:\n", y_pred)

# # Assuming you need to save predictions to a CSV file
# test_data['purchase_prediction'] = y_pred
# test_data[['user_id', 'purchase_prediction']].to_csv('/home/shubham/2021_6sense_DS_Takehome_Challenge/test_predictions.csv', index=False)

import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
# Load data
data = pd.read_csv('/home/shubham/2021_6sense_DS_Takehome_Challenge/training.tsv', delimiter='\t', header=None, names=['user_id', 'date', 'activity'])
data_test = pd.read_csv('/home/shubham/2021_6sense_DS_Takehome_Challenge/test.tsv', delimiter='\t', header=None, names=['user_id', 'date', 'activity'])

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data_test['date'] = pd.to_datetime(data_test['date'], errors='coerce')

# Convert activity column to lowercase
data['activity'] = data['activity'].str.lower()
data_test['activity'] = data_test['activity'].str.lower()

# Extract temporal features
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['day_of_month'] = data['date'].dt.day

data_test['day_of_week'] = data_test['date'].dt.dayofweek
data_test['month'] = data_test['date'].dt.month
data_test['day_of_month'] = data_test['date'].dt.day

# Calculate time since last activity
data['time_since_last_activity'] = data.groupby('user_id')['date'].diff().dt.days.fillna(0)
data_test['time_since_last_activity'] = data_test.groupby('user_id')['date'].diff().dt.days.fillna(0)
data['time_since_last_activity'] = data['time_since_last_activity'].astype(int)
data_test['time_since_last_activity'] = data_test['time_since_last_activity'].astype(int)

# Create target variable: Purchase indicator
data['purchase'] = np.where(data['activity'] == 'purchase', 1, 0)

# Encode activities
label_encoder = LabelEncoder()
data['activity_encoded'] = label_encoder.fit_transform(data['activity'])
data_test['activity_encoded'] = label_encoder.transform(data_test['activity'])


# Aggregate data by user_id
user_agg = data.groupby('user_id').agg({
    'activity_encoded': lambda x: list(x),
    'day_of_week': lambda x: list(x),
    'month': lambda x: list(x),
    'day_of_month': lambda x: list(x),
    'time_since_last_activity': lambda x: list(x),
    'purchase': 'sum'
}).reset_index()

user_agg_test = data_test.groupby('user_id').agg({
    'activity_encoded': lambda x: list(x),
    'day_of_week': lambda x: list(x),
    'month': lambda x: list(x),
    'day_of_month': lambda x: list(x),
    'time_since_last_activity': lambda x: list(x)
}).reset_index()

# Feature Engineering: Activity counts, recent activity, etc.
user_agg['purchase_target'] = user_agg['purchase'].apply(lambda x: 1 if x > 0 else 0)

# Calculate the 90th percentile of sequence lengths
seq_lengths = user_agg['activity_encoded'].apply(len)
seq_length = int(np.percentile(seq_lengths, 90))

print(f"90th percentile sequence length: {seq_length}")

# Define sequence length and pad sequences
def pad_sequences(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded[i] = np.array(seq[:maxlen])
        else:
            padded[i, -len(seq):] = np.array(seq)
    return padded

# Separate the indices for class 0 and class 1
class_0_indices = np.where(user_agg['purchase_target'] == 0)[0]
class_1_indices = np.where(user_agg['purchase_target'] == 1)[0]

# Randomly undersample class 0 indices to match the number of class 1 indices
undersampled_class_0_indices = np.random.choice(class_0_indices, size=len(class_1_indices), replace=False)

# Combine the undersampled class 0 indices with class 1 indices
balanced_indices = np.concatenate([undersampled_class_0_indices, class_1_indices])

# Create balanced dataset
user_agg = user_agg.iloc[balanced_indices].reset_index(drop=True)

features = ['activity_encoded', 'time_since_last_activity', 'day_of_week', 'month', 'day_of_month']
scalers = {feature: StandardScaler() for feature in features}

# Fit and transform the training data
for feature in features:
    user_agg[feature] = list(scalers[feature].fit_transform(pad_sequences(user_agg[feature], seq_length).reshape(-1, 1)).reshape(len(user_agg), seq_length))

# Apply the same scaling to the test data
for feature in features:
    user_agg_test[feature] = list(scalers[feature].transform(pad_sequences(user_agg_test[feature], seq_length).reshape(-1, 1)).reshape(len(user_agg_test), seq_length))

# Concatenate all padded and scaled sequences for training data
X = np.stack([np.array(user_agg[feature].tolist()) for feature in features], axis=-1)

# Concatenate all padded and scaled sequences for test data
X_test = np.stack([np.array(user_agg_test[feature].tolist()) for feature in features], axis=-1)
y = user_agg['purchase_target'].values

# Reshape X to 3D for LSTM: (num_samples, seq_length, input_size)
X = X.reshape((X.shape[0], seq_length, X.shape[2]))
X_test = X_test.reshape((X_test.shape[0], seq_length, X_test.shape[2]))

# Split balanced data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)



# Update input size
input_size = X.shape[2]  # Updated input size based on concatenated features

# Define LSTM model with the updated input size
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Hyperparameters
hidden_size = 64
num_layers = 2
output_size = 1
num_epochs = 50
learning_rate = 0.00005


# Reinitialize model with the new input size
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Early stopping parameters
patience = 3
min_delta = 0.001
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total_train += y_batch.size(0)
        correct_train += (predicted == y_batch).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = correct_train / total_train

    # Validation
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_val += y_batch.size(0)
            correct_val += (predicted == y_batch).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct_val / total_val

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Early stopping check
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f'Early stopping triggered after {epoch + 1} epochs')
        break




# Predict on test data
model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    y_test_pred = model(X_test_tensor)
    y_test_pred = (y_test_pred.cpu().numpy() > 0.5).astype(int)

# Save predictions to CSV file
user_agg_test['purchase_prediction'] = y_test_pred
user_agg_test[['user_id', 'purchase_prediction']].to_csv('/home/shubham/2021_6sense_DS_Takehome_Challenge/test_predictions_LSTM.csv', index=False)

