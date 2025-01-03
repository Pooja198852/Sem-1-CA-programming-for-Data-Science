import pandas as pd
import plotly.express as px
import plotly.graph_objects as go# Load the dataset
file_path = 'dataset/dataset.txt'
# Read the data file
with open(file_path, 'r') as file:
    lines = file.readlines()
    # Filter and parse transaction data
    data = []
    current_type = None

    for line in lines:
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            current_type = line.split(" - ")[1]
        elif line.startswith("END LAUNDERING ATTEMPT"):
            current_type = None
        elif current_type and len(line.split(',')) == 11:
            parts = line.split(',')
            data.append(parts + [current_type])
            # Create a DataFrame
            columns = [
                'Timestamp', 'Source_ID', 'Source_Account', 'Destination_ID',
                'Destination_Account', 'Amount', 'Currency', 'Amount_Converted',
                'Currency_Converted', 'Transfer_Type', 'Flag', 'Type'
            ]
            df = pd.DataFrame(data, columns=columns)df.head()
            df.shape
            # Data Cleaning
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Convert to datetime
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')  # Convert to float
            df['Amount_Converted'] = pd.to_numeric(df['Amount_Converted'], errors='coerce')
            df['Flag'] = pd.to_numeric(df['Flag'], errors='coerce')
            # Drop rows with missing values
            df = df.dropna()
            df.shape
            # Basic EDA
            print("Dataset Info:")
            df.info()
            print("\nSummary Statistics:")
            df.describe()
            # Transaction Amount Distribution
            fig1 = px.histogram(df, x='Amount', title='Transaction Amount Distribution', nbins=50)
            fig1.update_layout(bargap=0.2)
            fig1.show()
            fig2 = px.line(df, x='Timestamp', y='Amount', title='Transaction Trends Over Time', color='Type')
            fig2.show()
            # Currency Distribution
            fig3 = px.pie(df, names='Currency', title='Currency Distribution', hole=0.3)
            fig3.show()
            # Source Accounts with the Most Transactions
            top_sources = df['Source_Account'].value_counts().head(10).reset_index()
            top_sources.columns = ['Source_Account', 'Transaction_Count']
            fig4 = px.bar(top_sources, x='Source_Account', y='Transaction_Count',
                          title='Top Source Accounts by Transaction Count')
            fig4.show()
            # Amount by Transfer Type
            fig5 = px.box(df, x='Transfer_Type', y='Amount', title='Transaction Amounts by Transfer Type')
            fig5.show()
            df.to_csv('dataset/cleaned_dataset.csv', index=False)


            import pandas as pd
            import numpy as np
            import plotly.express as px
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report, confusion_matrix
            from xgboost import XGBClassifier
            from sklearn.preprocessing import LabelEncoder

            # Load the cleaned dataset
            df = pd.read_csv('dataset/cleaned_dataset.csv')
            # Convert timestamp to datetime
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            # Objective: Daily Transaction Volume Analysis
            df['Date'] = df['Timestamp'].dt.date
            daily_volume = df.groupby('Date')['Amount'].sum().reset_index()
            # Plot daily transaction volume
            fig1 = px.line(daily_volume, x='Date', y='Amount', title='Daily Transaction Volume')
            fig1.show()
            # Objective: Currency-wise Transaction Distribution
            currency_distribution = df.groupby('Currency')['Amount'].sum().reset_index()
            # Visualize currency-wise distribution
            fig2 = px.pie(currency_distribution, names='Currency', values='Amount',
                          title='Currency-Wise Transaction Distribution')
            fig2.show()
            # Transfer Type-wise Transaction Analysis
            transfer_type_distribution = df.groupby('Transfer_Type')['Amount'].sum().reset_index()
            # Visualize transfer type distribution
            fig3 = px.bar(transfer_type_distribution, x='Transfer_Type', y='Amount',
                          title='Transfer Type-wise Transaction Distribution')
            fig3.show()
            # Objective: Apply Data Mining and Machine Learning
            # Feature Engineering
            df['Hour'] = df['Timestamp'].dt.hour
            df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
            df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            # Encode categorical variables using LabelEncoder
            label_encoders = {}
            categorical_columns = ['Source_ID', 'Source_Account', 'Destination_ID', 'Destination_Account']

            for col in categorical_columns:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col])
                # Encode other categorical columns
                df['Currency'] = df['Currency'].astype('category').cat.codes
                df['Transfer_Type'] = df['Transfer_Type'].astype('category').cat.codes
                df['Type'] = df['Type'].astype('category').cat.codes
                # Define features and target
                features = ['Source_ID', 'Source_Account', 'Destination_ID', 'Destination_Account',
                            'Amount', 'Currency', 'Transfer_Type', 'Hour', 'DayOfWeek', 'IsWeekend']
                X = df[features]
                y = df['Flag']  # Target variable
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                # Random Forest Classifier
                rf_model = RandomForestClassifier(random_state=42)
                rf_model.fit(X_train, y_train)
                y_pred_rf = rf_model.predict(X_test)
                print("\nRandom Forest Classifier Report:")
                print(classification_report(y_test, y_pred_rf))
                print("Class distribution in the dataset:")
                print(y.value_counts())
                import pandas as pd
                import random
                import uuid
                from datetime import datetime, timedelta


                # Function to generate synthetic transactions with noise
                def generate_transaction(flag, num_rows=1000):
                    transactions = []
                    for _ in range(num_rows):
                        timestamp = datetime(2022, 8, 1) + timedelta(
                            minutes=random.randint(0, 30 * 24 * 60)  # Randomize within a month
                        )
                        source_id = random.randint(1000, 99999)
                        source_account = str(uuid.uuid4())[:10].upper()
                        destination_id = random.randint(1000, 99999)
                        destination_account = str(uuid.uuid4())[:10].upper()

                        # Add randomness to amounts and overlap patterns between classes
                        if flag == 1:
                            amount = round(random.uniform(30000.0, 100000.0), 2) + random.uniform(-5000,
                                                                                                  5000)  # Add noise
                            currency = random.choice(["Euro", "Ruble", "US Dollar"])  # Overlap currency choices
                            transfer_type = random.choice(["RTGS", "SEPA", "Wire"])
                        else:
                            amount = round(random.uniform(10.0, 40000.0), 2) + random.uniform(-2000, 2000)  # Add noise
                            currency = random.choice(["US Dollar", "Yuan", "Euro"])  # Overlap currency choices
                            transfer_type = random.choice(["ACH", "Wire", "SEPA"])

                        txn_type = random.choice(["STACK", "CYCLE: Max 12 hops", "FAN-IN"])

                        transactions.append(
                            [
                                timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                source_id,
                                source_account,
                                destination_id,
                                destination_account,
                                max(0, amount),  # Ensure no negative amounts
                                currency,
                                max(0, amount),  # Amount converted (with noise)
                                currency,
                                transfer_type,
                                flag,
                                txn_type,
                            ]
                        )
                    return transactions


                # Number of rows per class
                num_rows = 20000

                # Generate synthetic data
                synthetic_flagged = generate_transaction(flag=1, num_rows=num_rows)
                synthetic_non_flagged = generate_transaction(flag=0, num_rows=num_rows)

                # Combine and shuffle dataset
                columns = [
                    "Timestamp",
                    "Source_ID",
                    "Source_Account",
                    "Destination_ID",
                    "Destination_Account",
                    "Amount",
                    "Currency",
                    "Amount_Converted",
                    "Currency_Converted",
                    "Transfer_Type",
                    "Flag",
                    "Type",
                ]
                synthetic_data = pd.DataFrame(synthetic_flagged + synthetic_non_flagged, columns=columns)
                synthetic_data = synthetic_data.sample(frac=1, random_state=42).reset_index(drop=True)

                # Randomly flip 5% of the labels to simulate misclassification
                flip_indices = synthetic_data.sample(frac=0.05, random_state=42).index
                synthetic_data.loc[flip_indices, "Flag"] = 1 - synthetic_data.loc[flip_indices, "Flag"]

                # Save the dataset
                synthetic_data.to_csv("dataset/synthetic_dataset.csv", index=False)

                print("Noisy synthetic dataset created with balanced classes:")
                print(synthetic_data["Flag"].value_counts())
                # Load the synthetic dataset
                df = pd.read_csv("dataset/synthetic_dataset.csv")
                # Convert timestamp to datetime
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                # Feature Engineering
                df["Hour"] = df["Timestamp"].dt.hour
                df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
                df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)
                # Encode categorical variables
                label_encoders = {}
                categorical_columns = ["Source_ID", "Source_Account", "Destination_ID", "Destination_Account"]

                for col in categorical_columns:
                    label_encoders[col] = LabelEncoder()
                    df[col] = label_encoders[col].fit_transform(df[col])

                df["Currency"] = df["Currency"].astype("category").cat.codes
                df["Transfer_Type"] = df["Transfer_Type"].astype("category").cat.codes
                df["Type"] = df["Type"].astype("category").cat.codes
                # Define features and target
                features = [
                    "Source_ID",
                    "Destination_ID",
                    "Amount",
                    "Currency",
                    "Transfer_Type",
                    "Hour",
                    "DayOfWeek",
                    "IsWeekend",
                ]
                X = df[features]
                y = df["Flag"]
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                # Random Forest Classifier
                rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, min_samples_split=10)
                rf_model.fit(X_train, y_train)
                y_pred_rf = rf_model.predict(X_test)
                print("\nRandom Forest Classifier Report:")
                print(classification_report(y_test, y_pred_rf))
                # Gradient Boosting Classifier
                from sklearn.ensemble import GradientBoostingClassifier

                gb_model = GradientBoostingClassifier(random_state=42, n_estimators=200, max_depth=5, learning_rate=0.1)
                gb_model.fit(X_train, y_train)
                y_pred_gb = gb_model.predict(X_test)
                print("\nGradient Boosting Classifier Report:")
                print(classification_report(y_test, y_pred_gb))
                import pandas as pd
                import numpy as np


                # Function to handle unseen labels for LabelEncoder
                def transform_with_unseen_handling(label_encoder, column_data):
                    # Get the current unique classes from the encoder
                    unique_classes = list(label_encoder.classes_)
                    # Add any unseen labels to the encoder's classes
                    for item in column_data:
                        if item not in unique_classes:
                            unique_classes.append(item)
                    # Update the encoder's classes
                    label_encoder.classes_ = np.array(unique_classes)
                    # Transform the data
                    return label_encoder.transform(column_data)


                # Create sample data
                sample_data = pd.DataFrame({
                    'Source_ID': [12345, 54321],
                    'Destination_ID': [67890, 98765],
                    'Amount': [60000, 15000],  # One large and one small transaction
                    'Currency': ['Euro', 'US Dollar'],  # Different currencies
                    'Transfer_Type': ['RTGS', 'Wire'],  # Specific transfer types
                    'Hour': [14, 22],  # Different times of the day
                    'DayOfWeek': [2, 5],  # One weekday and one weekend day
                    'IsWeekend': [0, 1]  # Flag weekend
                })

                # Encode sample data using the same encoders as training data
                sample_data['Currency'] = sample_data['Currency'].astype('category').cat.codes
                sample_data['Transfer_Type'] = sample_data['Transfer_Type'].astype('category').cat.codes

                # Handle unseen labels for categorical columns
                for col in ['Source_ID', 'Destination_ID']:
                    if col in label_encoders:
                        sample_data[col] = transform_with_unseen_handling(label_encoders[col], sample_data[col])

                # Make predictions using Random Forest
                rf_predictions = rf_model.predict(sample_data)
                print("\nRandom Forest Predictions:")
                for idx, pred in enumerate(rf_predictions):
                    print(f"Sample {idx + 1}: {'Flagged' if pred == 1 else 'Not Flagged'}")

                # Make predictions using Gradient Boosting
                gb_predictions = gb_model.predict(sample_data)
                print("\nGradient Boosting Predictions:")
                for idx, pred in enumerate(gb_predictions):
                    print(f"Sample {idx + 1}: {'Flagged' if pred == 1 else 'Not Flagged'}")