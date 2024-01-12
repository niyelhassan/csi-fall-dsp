import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import os
import time
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Load processed data
#data_files = ['pd_nf_raw', 'pd_nf_time', 'pd_nf_freq', 'pd_nf_time_n_freq', "pd_bw_raw", "pd_bw_time", "pd_bw_freq", "pd_bw_time_n_freq", "pd_dwt_raw", "pd_dwt_time", "pd_dwt_freq", "pd_dwt_time_n_freq", "pd_ham_raw", "pd_ham_time", "pd_ham_freq", "pd_ham_time_n_freq"]
data_files = ['pd_bw_freq', 'pd_bw_raw', 'pd_bw_time_n_freq', 'pd_bw_time', 'pd_dwt_freq', 'pd_dwt_raw', 'pd_dwt_time_n_freq', 'pd_dwt_time', 'pd_ham_freq', 'pd_ham_raw', 'pd_ham_time_n_freq', 'pd_ham_time', 'pd_nf_freq', 'pd_nf_raw', 'pd_nf_time_n_freq', 'pd_nf_time']

def build_model():
    print("Building the RandomForest model.")
    model = RandomForestClassifier(verbose=2, n_estimators=500)
    print("Model built successfully.")
    return model

# Rest of the code remains the same...
# Check if the CSV file exists and load it if it does
csv_file = 'rf_final'
if os.path.exists("metrics_" + csv_file + '.csv'):
    all_metrics = pd.read_csv("metrics_" + csv_file + '.csv')
else:
    all_metrics = pd.DataFrame(columns=['data_file', 'run', 'precision', 'recall', 'f1-score', 'accuracy', ])

try:
    for data_file in data_files:
        # Load processed data
        data = np.load(f'{data_file}.npz')
        x_train = data['x_train']
        x_valid = data['x_valid']
        y_train = data['y_train']
        y_valid = data['y_valid']



        for i in range(1,6):  # Train and test the model 3 times
            # Check if this model has already been trained
            if ((all_metrics['data_file'] == data_file) & (all_metrics['run'] == i)).any():
                print(f"Skipping {data_file} run {i} because it's already been trained.")
                continue

            x_all = np.concatenate([x_train, x_valid], axis=0)
            y_all = np.concatenate([y_train, y_valid], axis=0)

            print(x_all.shape, y_all.shape)


            # Split the data into training and validation sets
            x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, train_size=0.9, random_state=i, stratify=y_all)



# reshape data to 2D if it is 3D
            if len(x_train.shape) == 3:
                x_train = x_train.reshape(x_train.shape[0], -1)
                x_valid = x_valid.reshape(x_valid.shape[0], -1)


            # Replace infinities with the maximum finite float64 number
            x_train[~np.isfinite(x_train)] = np.finfo(np.float32).max
            x_valid[~np.isfinite(x_valid)] = np.finfo(np.float32).max
            from sklearn.impute import SimpleImputer

            # Create an imputer object that replaces NaN values with the mean value of the column
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

            # Fit the imputer to the training data
            imputer.fit(x_train)

            # Transform the training and validation data
            x_train = imputer.transform(x_train)
            x_valid = imputer.transform(x_valid)



            # Build the model
            model = build_model()

            # Train the model
            print("Starting training process.")
            start_time = time.time()
            model.fit(x_train, np.argmax(y_train, axis=1))
            end_time = time.time()
            print("Training complete.")
            print(f"Training time: {(end_time - start_time) / 60} minutes")

            # Rest of the code remains the same...

            # Evaluate the model
            accuracy = model.score(x_valid, np.argmax(y_valid, axis=1))


            # Compute AUC
            y_pred_proba = model.predict_proba(x_valid)
           
            #print accuracy
            print(f"Accuracy: {accuracy * 100:.2f}%")

            # Compute confusion matrix
            # Compute confusion matrix
            y_true = np.argmax(y_valid, axis=1)
            y_pred = np.argmax(y_pred_proba, axis=1)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            # Extract 'fall' class performance
            labels = ('bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk')
            # Compute confusion matrix for 'fall' class
            fall_class_index = labels.index('fall')  # Get the index of 'fall' class
            y_true_fall = (y_true == fall_class_index).astype(int)  # 1 if 'fall', 0 otherwise
            y_pred_fall = (y_pred == fall_class_index).astype(int)  # 1 if 'fall', 0 otherwise
            tn, fp, fn, tp = confusion_matrix(y_true_fall, y_pred_fall).ravel()

            
            fall_report_dict = classification_report(np.argmax(y_valid, axis=1), np.argmax(y_pred_proba, axis=1), target_names=labels, output_dict=True)
            fall_metrics = {metric: fall_report_dict['fall'][metric] for metric in ['precision', 'recall', 'f1-score']}
            
            # Compute metrics based on confusion matrix
            precision_calc = tp / (tp + fp)
            recall_calc = tp / (tp + fn)
            specificity_calc = tn / (tn + fp)
            accuracy_calc = (tp + tn) / (tp + tn + fp + fn)

            # After training
            fall_metrics.update({
                'data_file': data_file, 
                'run': i, 
                'accuracy': accuracy, 
                'tp': tp, 
                'fp': fp, 
                'tn': tn, 
                'fn': fn,
                'precision_calc': precision_calc,
                'recall_calc': recall_calc,
                'specificity_calc': specificity_calc,
                'accuracy_calc': accuracy_calc
            })

            # Append the metrics for this run to the DataFrame
            fall_metrics_df = pd.DataFrame([fall_metrics])  
            all_metrics = pd.concat([all_metrics, fall_metrics_df], ignore_index=True)

            # Save the DataFrame to a CSV file after each model
            all_metrics.to_csv("metrics_" + csv_file + ".csv", index=False)
            print(all_metrics)

    # Calculate the average metrics for each data file
    avg_metrics = all_metrics.groupby('data_file').mean().reset_index()

    # Add the number of runs for each data file
    avg_metrics['runs'] = all_metrics.groupby('data_file').size().values

    # Drop the 'run' column
    avg_metrics = avg_metrics.drop(columns=['run'])

    # Make 'runs' the second column
    cols = avg_metrics.columns.tolist()
    cols.insert(1, cols.pop(cols.index('runs')))
    avg_metrics = avg_metrics[cols]

    # Convert precision, recall, f1, auc, accuracy to percentages
    for column in ['precision', 'recall', 'f1-score', 'accuracy']:
        avg_metrics[column] = avg_metrics[column].apply(lambda x: f"{x * 100:.2f}%")


    # Save the average metrics to a CSV file
    avg_metrics.to_csv("calc_" + csv_file + ".csv", index=False)
    print("Average metrics saved to CSV file.")
    
except Exception as e:
    with open("error_log.txt", "a") as file:
        file.write(f"An error occurred: {e}\n")
    raise  # Re-raise the exception for further handling if necessary
