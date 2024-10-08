import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

class DataLoader:
    def __init__(self, csv_file_path='/mnt/data/instagram_data.csv'):
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

    def get_image_dataset(self, image_dir='/mnt/data/images', batch_size=32, image_size=(224, 224)):
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

    def summary(self):
        # Display basic statistics about the data
        print("Dataset Summary:")
        print(f"Total records: {len(self.data)}")
        print(f"High engagement posts: {(self.data['likes'] > 10000).sum()}")
        print(f"Low engagement posts: {(self.data['likes'] <= 10000).sum()}")
        print(self.data.describe())

if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader.summary()
    image_dataset = data_loader.get_image_dataset()
    print(f"Loaded image dataset: {image_dataset}")