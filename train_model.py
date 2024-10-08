import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import image_dataset_from_directory

# Set global policy for mixed precision
mixed_precision.set_global_policy('mixed_float16')

class DataLoader:
    def __init__(self, csv_file_path):
        # Load CSV data into a pandas dataframe
        self.data = pd.read_csv(csv_file_path)
        self.images_path = self.data['imageUrl'].tolist()
        self.labels = self.data['likes'].apply(self._classify_likes).tolist()

    def _classify_likes(self, likes):
        # Classify the post based on the number of likes (for demonstration purposes)
        if likes > 10000:
            return 1  # High engagement
        else:
            return 0  # Low engagement

    def get_image_dataset(self, image_dir, batch_size=32, image_size=(224, 224)):
        # Create a dataset from images stored in a directory
        dataset = image_dataset_from_directory(
            image_dir,
            labels=self.labels,
            label_mode='int',
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True
        )
        return dataset

class TrainModel:
    def __init__(self, csv_file_path, image_dir):
        # Load dataset using DataLoader
        self.data_loader = DataLoader(csv_file_path)
        self.train_dataset = self.data_loader.get_image_dataset(image_dir)

    def build_model(self, input_shape=(224, 224, 3)):
        # Build a transfer learning model using MobileNetV2
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        base_model.trainable = False
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, epochs=10):
        # Build the model
        model = self.build_model()

        # Define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
        lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr * tf.math.exp(-0.1) if epoch > 5 else lr)

        # Train the model
        model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.train_dataset,  # Placeholder, should use a proper validation set
            callbacks=[early_stopping, model_checkpoint, lr_scheduler]
        )

if __name__ == "__main__":
    # Training Section
    csv_path = '/mnt/data/instagram_data.csv'  # Use the updated dataset path
    image_dir = 'images/'  # Replace with correct path to images directory
    trainer = TrainModel(csv_path, image_dir)
    trainer.train(epochs=10)