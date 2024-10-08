import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_loader import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class EvaluateModel:
    def __init__(self, csv_file_path='/mnt/data/instagram_data.csv', image_dir='/mnt/data/images', model_path='best_model.h5'):
        # Load dataset using DataLoader
        self.data_loader = DataLoader(csv_file_path)
        self.test_dataset = self.data_loader.get_image_dataset(image_dir)
        # Load pre-trained model
        self.model = load_model(model_path)

    def evaluate(self):
        # Evaluate the model on the test dataset
        loss, accuracy = self.model.evaluate(self.test_dataset)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    def predict(self, num_batches=1):
        # Predict the classes of images in the test dataset
        y_true = []
        y_pred = []

        for images, labels in self.test_dataset.take(num_batches):
            predictions = self.model.predict(images)
            predicted_classes = (predictions > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted_classes.flatten())

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")

    def plot_confusion_matrix(self, num_batches=1):
        y_true = []
        y_pred = []

        for images, labels in self.test_dataset.take(num_batches):
            predictions = self.model.predict(images)
            predicted_classes = (predictions > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted_classes.flatten())

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

if __name__ == "__main__":
    csv_path = '/mnt/data/instagram_data.csv'  # Use the updated dataset path
    image_dir = '/mnt/data/images'  # Replace with correct path to images directory
    model_path = 'best_model.h5'  # Path to the trained model
    evaluator = EvaluateModel(csv_path, image_dir, model_path)
    evaluator.evaluate()
    evaluator.predict()
    evaluator.plot_confusion_matrix()

