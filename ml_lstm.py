from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd
import os
import time
from sklearn.metrics import confusion_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Load processed data
data_files = ['pd_nf_raw_lstm', "pd_nf_time_lstm", "pd_nf_freq_lstm", "pd_nf_time_n_freq_lstm", "pd_bw_raw_lstm", "pd_bw_time_lstm", "pd_bw_freq_lstm", "pd_bw_time_n_freq_lstm", "pd_dwt_raw_lstm", "pd_dwt_time_lstm", "pd_dwt_freq_lstm", "pd_dwt_time_n_freq_lstm", "pd_ham_raw_lstm", "pd_ham_time_lstm", "pd_ham_freq_lstm", "pd_ham_time_n_freq_lstm"]
data_files = ['pd_raw_data_2_sets']
metric_file_name = 'lstm-2-sets'
    
# Callback for logging purposes
class PrintCallback(Callback):
    def __init__(self, data_file, run):
        self.data_file = data_file
        self.run = run
        self.logs = {}

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
        print(f"Starting training for {self.data_file} - Run: {self.run}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        logs['epoch_time'] = epoch_time
        print(f"{self.data_file} - Run: {self.run} - Epoch: {epoch+1}/50 - Accuracy: {logs['accuracy']:.3f} - Time: {epoch_time:.3f} seconds")

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        average_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        self.logs['total_time'] = total_time
        self.logs['average_epoch_time'] = average_epoch_time
        print(f"Training for {self.data_file} - Run: {self.run} completed in {total_time:.3f} seconds. Average epoch time: {average_epoch_time:.3f} seconds")


# Function to construct the LSTM model
# def build_model(input_shape, n_units_lstm=200, n_labels=7):
#     print(f"Building the LSTM model with {n_units_lstm} LSTM units.")
#     inputs = Input(shape=input_shape)
#     lstm_out = LSTM(n_units_lstm, return_sequences=False)(inputs)
#     outputs = Dense(n_labels, activation='softmax')(lstm_out)

#     model = Model(inputs, outputs)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     print("Model built successfully.")
#     return model

def build_model(input_shape, n_units_lstm=200, n_labels=7):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(n_units_lstm)(inputs)
    outputs = Dense(n_labels, activation='softmax')(lstm_out)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# CSV file for storing metrics
csv_file = "metrics_" + metric_file_name + '.csv'
if os.path.exists(csv_file):
    all_metrics = pd.read_csv(csv_file)
else:
    all_metrics = pd.DataFrame(columns=['data_file', 'run', 'precision', 'recall', 'f1-score', 'auc', 'accuracy', 'loss', 'total_time', 'average_epoch_time'])

try:
    for data_file in data_files:
        # Load processed data
        # data = np.load(f'{data_file}.npz')
        # x_train, x_valid, x_test = data['x_train'], data['x_valid'], data['x_test']
        # y_train, y_valid, y_test = data['y_train'], data['y_valid'], data['y_test']

            # Load processed data
        data = np.load(f'{data_file}.npz')
        x_train, x_valid = data['x_train'], data['x_valid']
        y_train, y_valid = data['y_train'], data['y_valid']

        
        # if x_train has only two dimensions Add an extra dimension to x_train, x_valid and x_test
        if len(x_train.shape) == 2:
            x_train = tf.expand_dims(x_train, axis=-1)
            x_valid = tf.expand_dims(x_valid, axis=-1)
            #x_test = tf.expand_dims(x_test, axis=-1)

        # Training the model
        input_shape = (x_train.shape[1], x_train.shape[2])
        for i in range(1, 4):  # Train and test the model 3 times
            # Skip if this model has already been trained
            if ((all_metrics['data_file'] == data_file) & (all_metrics['run'] == i)).any():
                print(f"Skipping {data_file} run {i} because it's already been trained.")
                continue

            
            model = build_model(input_shape)
            print_callback = PrintCallback(data_file, i)
            model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=50, batch_size=128, verbose=0, callbacks=[print_callback])
            print("Training complete.")

            # Evaluate the model on the test set
            #loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

            loss, accuracy = model.evaluate(x_valid, y_valid, verbose=0)


            # # Compute AUC on the test set
            # y_pred_proba = model.predict(x_test)
            # auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            

            # Compute AUC on the test set
            y_pred_proba = model.predict(x_valid)
            auc = roc_auc_score(y_valid, y_pred_proba, multi_class='ovr')


            # Compute confusion matrix on the test set
            # y_true = np.argmax(y_test, axis=1)
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
                'loss': loss, 
                'auc': auc, 
                'total_time': print_callback.logs['total_time'], 
                'average_epoch_time': print_callback.logs['average_epoch_time'], 
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
            all_metrics.to_csv(csv_file, index=False)
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
    for column in ['precision', 'recall', 'f1-score', 'auc', 'accuracy']:
        avg_metrics[column] = avg_metrics[column].apply(lambda x: f"{x * 100:.2f}%")

    # Convert total_time and average_epoch_time to mm:ss:ms format
    for column in ['total_time', 'average_epoch_time']:
        avg_metrics[column] = avg_metrics[column].apply(lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}:{int((x * 1000) % 1000):03d}")

    # Save the average metrics to a CSV file
    avg_metrics.to_csv("calc_" + metric_file_name + ".csv", index=False)
    print("Average metrics saved to CSV file.")
    
except Exception as e:
    with open("error_log.txt", "a") as file:
        file.write(f"An error occurred: {e}\n")
    raise  # Re-raise the exception for further handling if necessary
