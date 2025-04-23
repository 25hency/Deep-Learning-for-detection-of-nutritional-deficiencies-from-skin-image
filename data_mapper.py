import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import re
import warnings
warnings.filterwarnings("ignore")

class NHANESImageMapper:
    """
    Utility class to map NHANES patient IDs with dermatological images
    and create paired datasets for multi-modal learning.
    """
    def __init__(self, nhanes_dir, image_dir):
        """
        Initialize the mapper with paths to NHANES data and image directories.
        
        Args:
            nhanes_dir (str): Directory containing NHANES CSV files
            image_dir (str): Directory containing dermatological images
        """
        self.nhanes_dir = nhanes_dir
        self.image_dir = image_dir
        self.image_categories = []
        self.patient_image_mapping = {}
        self.dataset_mapping = {}
        
    def load_nhanes_data(self):
        """
        Load and merge relevant NHANES datasets.
        
        Returns:
            pandas.DataFrame: Merged NHANES data
        """
        # Load datasets with specified encoding to handle potential UnicodeDecodeError
        diet_data = pd.read_csv(os.path.join(self.nhanes_dir, 'diet.csv'), encoding='ISO-8859-1')
        examination_data = pd.read_csv(os.path.join(self.nhanes_dir, 'examination.csv'), encoding='ISO-8859-1')
        labs_data = pd.read_csv(os.path.join(self.nhanes_dir, 'labs.csv'), encoding='ISO-8859-1')
        demographic_data = pd.read_csv(os.path.join(self.nhanes_dir, 'demographic.csv'), encoding='ISO-8859-1')
        
        # Print the column names for debugging
        print("Diet data columns:", diet_data.columns[:10].tolist(), "...")
        
        # Extract only necessary columns - using columns that are actually in the dataset
        # We'll select SEQN (ID) and other common nutritional measurements
        diet_features = diet_data[['SEQN']]
        
        # Add additional diet columns if they exist
        diet_columns_to_try = ['DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TTFAT', 
                              'DR1DAY', 'DR1DBIH', 'WTDRD1', 'WTDR2D']
        existing_diet_cols = [col for col in diet_columns_to_try if col in diet_data.columns]
        if existing_diet_cols:
            diet_features = diet_data[['SEQN'] + existing_diet_cols]
            print(f"Using diet columns: {existing_diet_cols}")
        else:
            print("Warning: No matching dietary columns found. Using only SEQN from diet data.")
        
        # Lab features - check if they exist before using them
        lab_columns_to_try = ['LBXSCH', 'LBDHDL', 'LBXTC', 'LBXGLU', 'LBXSGL']
        existing_lab_cols = [col for col in lab_columns_to_try if col in labs_data.columns]
        lab_features = labs_data[['SEQN']]
        if existing_lab_cols:
            lab_features = labs_data[['SEQN'] + existing_lab_cols]
            print(f"Using lab columns: {existing_lab_cols}")
        else:
            print("Warning: No matching lab columns found. Using only SEQN from lab data.")
        
        # Examination features
        exam_columns_to_try = ['BMXBMI', 'BPXSY1', 'BPXDI1', 'BMXWT', 'BMXHT']
        existing_exam_cols = [col for col in exam_columns_to_try if col in examination_data.columns]
        exam_features = examination_data[['SEQN']]
        if existing_exam_cols:
            exam_features = examination_data[['SEQN'] + existing_exam_cols]
            print(f"Using examination columns: {existing_exam_cols}")
        else:
            print("Warning: No matching examination columns found. Using only SEQN from examination data.")
        
        # Demographic features
        demo_columns_to_try = ['RIAGENDR', 'RIDAGEYR', 'RIDRETH1', 'DMDEDUC2']
        existing_demo_cols = [col for col in demo_columns_to_try if col in demographic_data.columns]
        demo_features = demographic_data[['SEQN']]
        if existing_demo_cols:
            demo_features = demographic_data[['SEQN'] + existing_demo_cols]
            print(f"Using demographic columns: {existing_demo_cols}")
        else:
            print("Warning: No matching demographic columns found. Using only SEQN from demographic data.")
        
        # Merge datasets on SEQN (patient ID)
        merged_data = diet_features.merge(lab_features, on='SEQN', how='inner')
        merged_data = merged_data.merge(exam_features, on='SEQN', how='inner')
        merged_data = merged_data.merge(demo_features, on='SEQN', how='inner')
        
        print(f"Merged data shape: {merged_data.shape}")
        print(f"Merged data columns: {merged_data.columns.tolist()}")
        
        # Handle missing values - impute with mean for numerical features
        for col in merged_data.columns:
            if col != 'SEQN' and merged_data[col].dtype in [np.float64, np.int64]:
                merged_data[col] = merged_data[col].fillna(merged_data[col].mean())
        
        self.nhanes_data = merged_data
        return merged_data
    
    def scan_image_categories(self):
        """
        Scan the image directory to identify image categories.
        
        Returns:
            list: List of image category names
        """
        train_dir = os.path.join(self.image_dir, 'train')
        categories = [d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))]
        
        self.image_categories = categories
        return categories
    
    def create_synthetic_mapping(self, save_path=None):
        """
        Create a synthetic mapping between NHANES patient IDs and dermatological images.
        In a real-world scenario, this would be based on actual patient records.
        
        Args:
            save_path (str, optional): Path to save the mapping as JSON
            
        Returns:
            dict: Mapping between patient IDs and image paths
        """
        # Ensure NHANES data is loaded
        if not hasattr(self, 'nhanes_data'):
            self.load_nhanes_data()
            
        # Ensure image categories are scanned
        if not self.image_categories:
            self.scan_image_categories()
            
        # Get all patient IDs
        patient_ids = self.nhanes_data['SEQN'].unique().tolist()
        
        # Create a mapping dictionary
        patient_image_mapping = {}
        
        # Scan train and test directories to get all image paths
        train_images = {}
        test_images = {}
        
        for category in self.image_categories:
            train_category_dir = os.path.join(self.image_dir, 'train', category)
            test_category_dir = os.path.join(self.image_dir, 'test', category)
            
            if os.path.exists(train_category_dir):
                train_images[category] = [
                    os.path.join('train', category, img) for img in os.listdir(train_category_dir)
                    if img.endswith(('.jpg', '.jpeg', '.png'))
                ]
                
            if os.path.exists(test_category_dir):
                test_images[category] = [
                    os.path.join('test', category, img) for img in os.listdir(test_category_dir)
                    if img.endswith(('.jpg', '.jpeg', '.png'))
                ]
        
        # Assign images to patients based on a deterministic approach
        # This ensures reproducibility and consistent mapping
        np.random.seed(42)  # For reproducibility
        
        # Define condition-to-category mapping based on lab values
        # This is a simplified example - in a real scenario, this would be based on medical knowledge
        def determine_likely_condition(patient_data):
            """Determine likely condition based on patient data"""
            # Default values in case columns are missing
            bmi = 25.0
            cholesterol = 200.0
            age = 40
            gender = 0
            
            # Get values if columns exist
            if 'BMXBMI' in patient_data:
                bmi = patient_data['BMXBMI']
            if 'LBXSCH' in patient_data:
                cholesterol = patient_data['LBXSCH']
            if 'RIDAGEYR' in patient_data:
                age = patient_data['RIDAGEYR']
            if 'RIAGENDR' in patient_data:
                gender = patient_data['RIAGENDR']
            
            # Simplified mapping logic (this should be replaced with actual medical criteria)
            if bmi > 30:  # High BMI
                return ['Cellulitis Impetigo and other Bacterial Infections', 
                        'Psoriasis pictures Lichen Planus and related diseases']
            elif cholesterol > 240:  # High cholesterol
                return ['Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
                       'Light Diseases and Disorders of Pigmentation']
            elif 18 <= age <= 30:  # Young adults
                return ['Acne and Rosacea Photos', 'Eczema Photos']
            elif gender == 1:  # Male
                return ['Hair Loss Photos Alopecia and other Hair Diseases', 
                        'Nail Fungus and other Nail Disease']
            else:
                return ['Systemic Disease', 'Light Diseases and Disorders of Pigmentation']
        
        # Assign images to patients
        for patient_id in patient_ids:
            patient_data = self.nhanes_data[self.nhanes_data['SEQN'] == patient_id].iloc[0]
            
            # Determine likely conditions for this patient
            likely_conditions = determine_likely_condition(patient_data)
            
            # Filter categories to those that are available in our dataset
            available_conditions = [cond for cond in likely_conditions 
                                   if cond in self.image_categories]
            
            if available_conditions:
                # Choose one condition from the available ones
                selected_condition = np.random.choice(available_conditions)
                
                # Get available images for this condition
                available_images = (train_images.get(selected_condition, []) + 
                                   test_images.get(selected_condition, []))
                
                if available_images:
                    # Assign a random image to this patient
                    assigned_image = np.random.choice(available_images)
                    deficiency_label = self.image_categories.index(selected_condition)
                    
                    patient_image_mapping[patient_id] = {
                        'image_path': assigned_image,
                        'condition': selected_condition,
                        'deficiency_label': deficiency_label
                    }
        
        self.patient_image_mapping = patient_image_mapping
        
        # Save the mapping if a path is provided
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(patient_image_mapping, f, indent=4)
        
        return patient_image_mapping
    
    def create_paired_dataset(self, test_size=0.2, random_state=42):
        """
        Create paired datasets of NHANES data and images for training models.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing train and test datasets
        """
        # Ensure mapping exists
        if not self.patient_image_mapping:
            self.create_synthetic_mapping()
        
        # Create paired data
        paired_data = []
        
        for patient_id, img_info in self.patient_image_mapping.items():
            patient_data = self.nhanes_data[self.nhanes_data['SEQN'] == patient_id]
            
            if not patient_data.empty:
                # Extract tabular data without the SEQN
                tabular_features = patient_data.drop('SEQN', axis=1).iloc[0].to_dict()
                
                paired_data.append({
                    'patient_id': patient_id,
                    'tabular_data': tabular_features,
                    'image_path': os.path.join(self.image_dir, img_info['image_path']),
                    'condition': img_info['condition'],
                    'deficiency_label': img_info['deficiency_label']
                })
        
        # Split into train and test sets
        train_data, test_data = train_test_split(
            paired_data, test_size=test_size, random_state=random_state
        )
        
        self.dataset_mapping = {
            'train': train_data,
            'test': test_data,
            'num_classes': len(self.image_categories),
            'image_categories': self.image_categories
        }
        
        return self.dataset_mapping
    
    def get_dataset_stats(self):
        """
        Get statistics about the paired dataset.
        
        Returns:
            dict: Statistics about the dataset
        """
        if not self.dataset_mapping:
            raise ValueError("Dataset not created yet. Call create_paired_dataset() first.")
        
        stats = {
            'train_size': len(self.dataset_mapping['train']),
            'test_size': len(self.dataset_mapping['test']),
            'num_classes': self.dataset_mapping['num_classes'],
            'class_distribution': {}
        }
        
        # Count samples per class in train set
        train_distribution = {}
        for item in self.dataset_mapping['train']:
            condition = item['condition']
            train_distribution[condition] = train_distribution.get(condition, 0) + 1
        
        # Count samples per class in test set
        test_distribution = {}
        for item in self.dataset_mapping['test']:
            condition = item['condition']
            test_distribution[condition] = test_distribution.get(condition, 0) + 1
        
        stats['class_distribution']['train'] = train_distribution
        stats['class_distribution']['test'] = test_distribution
        
        return stats

# Usage example
if __name__ == "__main__":
    mapper = NHANESImageMapper(
        nhanes_dir="dataset/NHANES",
        image_dir="dataset/Image data"
    )
    
    # Load NHANES data
    nhanes_data = mapper.load_nhanes_data()
    print(f"Loaded NHANES data with {len(nhanes_data)} records")
    
    # Scan image categories
    categories = mapper.scan_image_categories()
    print(f"Found {len(categories)} image categories")
    
    # Create synthetic mapping
    mapping = mapper.create_synthetic_mapping(save_path="patient_image_mapping.json")
    print(f"Created mapping for {len(mapping)} patients")
    
    # Create paired dataset
    paired_dataset = mapper.create_paired_dataset()
    print(f"Created paired dataset with {len(paired_dataset['train'])} training samples and "
          f"{len(paired_dataset['test'])} testing samples")
    
    # Get dataset statistics
    stats = mapper.get_dataset_stats()
    print("Dataset statistics:")
    print(stats)