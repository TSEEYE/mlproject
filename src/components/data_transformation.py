import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import sys
import os
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        logging.info("Entered the Data Transformation Component")

        try:
            numerical_features = ['writing_score','reading_score']
            categorical_features = [
                'gender', 'race_ethnicity', 'parental_level_of_education', 'lunch','test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info("Numerical columns standard scaling completed")

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_endcoder', OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean = False))
                ]
            )

            logging.info(f'Categorical columns: {categorical_features}')
            logging.info(f"Numerical columns: {numerical_features}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline',cat_pipeline, categorical_features)
                    ]
                    )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer_object()

            target_column = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            input_features_train_df = train_df.drop(columns = [target_column], axis = 1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns = [target_column], axis = 1)
            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor_obj.transform(input_features_test_df)

            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_features_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessed object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)
        