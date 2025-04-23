import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class MultiModalDataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator for multi-modal learning with NHANES tabular data and dermatological images.
    
    This generator yields batches of paired tabular and image data for training multi-modal models.
    """
    def __init__(
        self,
        dataset,
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        augment=False,
        tabular_features=None,
        num_classes=None
    ):
        """
        Initialize the generator.
        
        Args:
            dataset (list): List of dictionaries containing paired data
            batch_size (int): Number of samples per batch
            image_size (tuple): Target size for images (height, width)
            shuffle (bool): Whether to shuffle data after each epoch
            augment (bool): Whether to apply data augmentation to images
            tabular_features (list): List of tabular features to use (if None, use all)
            num_classes (int): Number of classes for one-hot encoding (if None, will determine from data)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.num_classes = num_classes
        
        # Extract all tabular feature names if not specified
        if tabular_features is None and len(dataset) > 0:
            self.tabular_features = list(dataset[0]['tabular_data'].keys())
        else:
            self.tabular_features = tabular_features
            
        # Count samples and set up indexes
        self.n_samples = len(dataset)
        self.indexes = np.arange(self.n_samples)
        
        # Set up data augmentation if required
        if self.augment:
            self.img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            self.img_gen = None
            
        # Initialize data scalers
        self.init_scalers()
        
        # Determine number of classes if not provided
        if self.num_classes is None:
            all_labels = [s['deficiency_label'] for s in self.dataset]
            self.num_classes = max(all_labels) + 1
            print(f"Auto-detected {self.num_classes} classes from dataset labels")
        
        # Shuffle data initially if required
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def init_scalers(self):
        """
        Initialize scalers for tabular data normalization.
        
        This method creates and fits standard scalers for each numeric feature
        in the tabular data to ensure proper normalization.
        """
        # Extract all tabular data into a dataframe
        tabular_data_list = []
        for item in self.dataset:
            tabular_data_list.append(item['tabular_data'])
        
        df = pd.DataFrame(tabular_data_list)
        
        # Create scalers for numeric features
        self.scalers = {}
        for feature in self.tabular_features:
            if feature in df.columns and np.issubdtype(df[feature].dtype, np.number):
                scaler = StandardScaler()
                df[feature] = df[feature].fillna(df[feature].mean())  # Handle NaNs
                self.scalers[feature] = scaler.fit(df[[feature]])
    
    def __len__(self):
        """Return the number of batches per epoch."""
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Get batch at index `idx`.
        
        Args:
            idx (int): Batch index
            
        Returns:
            tuple: ([tabular_batch, image_batch], label_batch)
        """
        # Get batch indexes
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_samples = [self.dataset[i] for i in batch_indexes]
        
        # Generate data for this batch
        X_tabular, X_image, y = self.__data_generation(batch_samples)
        
        # Return inputs and outputs
        return [X_tabular, X_image], y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_samples):
        """
        Generate a batch of data.
        
        Args:
            batch_samples (list): List of sample dictionaries for this batch
            
        Returns:
            tuple: (X_tabular, X_image, y) containing input and output batches
        """
        # Initialize arrays
        batch_size = len(batch_samples)
        X_tabular = np.zeros((batch_size, len(self.tabular_features)))
        X_image = np.zeros((batch_size, self.image_size[0], self.image_size[1], 3))
        y = np.zeros((batch_size, 1))  # Assuming single-label classification
        
        # Fill the arrays
        for i, sample in enumerate(batch_samples):
            # Process tabular data
            for j, feature in enumerate(self.tabular_features):
                value = sample['tabular_data'].get(feature, 0)
                
                # Apply scaling if scaler exists for this feature
                if feature in self.scalers:
                    value = self.scalers[feature].transform([[value]])[0][0]
                
                X_tabular[i, j] = value
            
            # Process image data
            try:
                # Load and preprocess image
                img_path = sample['image_path']
                img = load_img(img_path, target_size=self.image_size)
                img_array = img_to_array(img)
                
                # Apply augmentation if needed
                if self.augment and self.img_gen:
                    img_array = self.img_gen.random_transform(img_array)
                
                # Apply EfficientNet preprocessing
                img_array = preprocess_input(img_array)
                
                X_image[i] = img_array
                
                # Get label
                y[i] = sample['deficiency_label']
            except Exception as e:
                print(f"Error processing image {sample['image_path']}: {str(e)}")
                # Use zeros for failed images
        
        # Convert labels to one-hot encoding using the consistent number of classes
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        return X_tabular, X_image, y_onehot


# Example usage
if __name__ == "__main__":
    # This is just a demonstration - in practice you would use your actual data
    from data_mapper import NHANESImageMapper
    
    # Create mapper and get paired dataset
    mapper = NHANESImageMapper(
        nhanes_dir="dataset/NHANES",
        image_dir="dataset/Image data"
    )
    
    paired_dataset = mapper.create_paired_dataset()
    
    # Create generators for training and testing
    train_generator = MultiModalDataGenerator(
        dataset=paired_dataset['train'],
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        augment=True
    )
    
    test_generator = MultiModalDataGenerator(
        dataset=paired_dataset['test'],
        batch_size=32,
        image_size=(224, 224),
        shuffle=False,
        augment=False
    )
    
    # Example of accessing a batch
    inputs, targets = train_generator[0]
    tabular_input, image_input = inputs
    
    print(f"Tabular input shape: {tabular_input.shape}")
    print(f"Image input shape: {image_input.shape}")
    print(f"Target shape: {targets.shape}")