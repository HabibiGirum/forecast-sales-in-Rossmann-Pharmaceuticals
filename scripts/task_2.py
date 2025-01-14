import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from scripts.task_1 import EDA
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Inherited class
class Task2(EDA):
    def __init__(self, data_paths):
        super().__init__(data_paths)
        self.train_df = None
        self.val_df = None

    def split_the_dataset(self):
        """Split data into training and validation sets."""
        self.test_merged_df['Date'] = pd.to_datetime(self.test_merged_df.Date)

        self.test_merged_df['Year'] = self.test_merged_df.Date.dt.year
        self.test_merged_df['Month'] = self.test_merged_df.Date.dt.month
        # self.test_merged_df['Day'] = self.test_merged_df.Date.dt.day

        print(f"Estimate Sales from {self.test_merged_df.Date.dt.date.min()} to {self.test_merged_df.Date.dt.date.max()}")
        self.test_merged_df
        self.train_df = self.reduced_train_df[self.reduced_train_df.Date.dt.year <= 2014]
        self.val_df = self.reduced_train_df[self.reduced_train_df.Date.dt.year == 2015]

        # print(f"Training Shape: {train_df.shape}")
        # print(f"Validation Shape: {val_df.shape}")
        print(f"Test Shape: {self.test_merged_df.shape}")
        logging.info(f"Training Shape: {self.train_df.shape}")
        logging.info(f"Validation Shape: {self.val_df.shape}")

    def data_preprocessing(self):
        """Preprocess train, validation, and test datasets."""
        input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'StoreType', 'Assortment', 'Promo2', 'Month', 'Year']
        target_col = 'Sales'

        # Replace categorical values with numeric ones for 'StateHoliday'
        self.train_df['StateHoliday'] = self.train_df['StateHoliday'].replace({'0': 0, 'a': 1, 'b': 2, 'c': 3}).astype(float)
        self.val_df['StateHoliday'] = self.val_df['StateHoliday'].replace({'0': 0, 'a': 1, 'b': 2, 'c': 3}).astype(float)
        self.test_merged_df['StateHoliday'] = self.test_merged_df['StateHoliday'].replace({'0': 0, 'a': 1, 'b': 2, 'c': 3}).astype(float)

        # Prepare inputs and targets
        self.train_inputs = self.train_df[input_cols].copy()
        self.train_targets = self.train_df[target_col].copy()
        self.val_inputs = self.val_df[input_cols].copy()
        self.val_targets = self.val_df[target_col].copy()
        self.test_inputs = self.test_merged_df[input_cols].copy()

        # One-hot encoding
        train_inputs = pd.get_dummies(self.train_inputs)
        val_inputs = pd.get_dummies(self.val_inputs)
        test_inputs = pd.get_dummies(self.test_inputs)

        # Align columns across datasets
        train_inputs, val_inputs = train_inputs.align(val_inputs, axis=1, fill_value=0)
        train_inputs, test_inputs = train_inputs.align(test_inputs, axis=1, fill_value=0)

        # Scale numerical columns
        num_cols = ['Store', 'DayOfWeek', 'Month', 'Year']
        scaler = MinMaxScaler().fit(train_inputs[num_cols])

        train_inputs[num_cols] = scaler.transform(train_inputs[num_cols])
        val_inputs[num_cols] = scaler.transform(val_inputs[num_cols])
        test_inputs[num_cols] = scaler.transform(test_inputs[num_cols])

        # Save processed inputs
        self.train_inputs = train_inputs
        self.val_inputs = val_inputs
        self.test_inputs = test_inputs

        
        
    def choose_model(self):
        """Train and evaluate models."""
        def rmspe(y_true, y_pred):
            # Ensure both arrays have the same length
            assert len(y_true) == len(y_pred)
            # Compute the percentage error for each observation
            percentage_error = (y_true - y_pred) / y_true
            
            # Exclude observations where true value is zero
            percentage_error[y_true == 0] = 0
            
            # Square the percentage errors
            squared_percentage_error = percentage_error ** 2
            
            # Compute the mean of the squared percentage errors
            mean_squared_percentage_error = np.mean(squared_percentage_error)
            
            # Compute the square root of the mean squared percentage error
            rmspe = np.sqrt(mean_squared_percentage_error)
            
            return rmspe # Convert to percentage

        def try_model(model):
            """Fit and evaluate a model."""
            model.fit(self.train_inputs, self.train_targets)

            train_preds = model.predict(self.train_inputs)
            val_preds = model.predict(self.val_inputs)

            # Get RMSE
            train_rmse = np.round(mean_squared_error(self.train_targets, train_preds, squared=False), 5)
            val_rmse = np.round(mean_squared_error(self.val_targets, val_preds, squared=False), 5)

            # Get RMSPE
            train_rmspe = np.round(rmspe(self.train_targets, train_preds), 5)
            val_rmspe = np.round(rmspe(self.val_targets, val_preds), 5)


            print(f"Train RMSE: {train_rmse}")
            print(f"Val RMSE: {val_rmse}")
            print()
            print(f"Train RMSPE: {train_rmspe}")
            print(f"Val RMSPE: {val_rmspe}")

            return model

        # Try different models
        random_forest = RandomForestRegressor(random_state=42, n_jobs=-1)
        best_model = try_model(random_forest)

        # Predict test data
        test_preds = best_model.predict(self.test_inputs)
        logging.info(f"Test predictions completed. Sample: {test_preds[:5]}")
        return test_preds

# # Main execution
# if __name__ == "__main__":
    