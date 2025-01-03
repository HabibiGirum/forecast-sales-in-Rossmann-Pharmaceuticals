import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EDA:
    def __init__(self, data_paths):
        """
        Initialize with data file paths.
        Args:
            data_paths (dict): Paths for train, test, store, and sample submission files.
        """
        self.data_paths = data_paths
        self.train = None
        self.test = None
        self.store = None
        self.sample_submission = None
        self.train_merged_df = None
        self.test_merged_df = None
        self.reduced_train_df = None
    
    def load_data(self):
        """Load datasets from specified file paths."""
        try:
            self.sample_submission = pd.read_csv(self.data_paths['sample_submission'])
            self.store = pd.read_csv(self.data_paths['store'])
            self.test = pd.read_csv(self.data_paths['test'])
            self.train = pd.read_csv(self.data_paths['train'])
            logging.info("Data loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Error loading data: {e}")
    
    def merge_data(self):
        """Merge store data with train and test datasets."""
        self.train_merged_df = pd.merge(self.train, self.store, on='Store', how='left').drop(['PromoInterval'], axis=1)
        self.test_merged_df = pd.merge(self.test, self.store, on='Store', how='left')
        logging.info("Data merged successfully.")
    
    def preprocess_data(self):
        """Handle null values and clean data."""
        self.train_merged_df['StateHoliday'].replace({'0': 0}, inplace=True)
        self.test_merged_df['StateHoliday'].replace({'0': 0}, inplace=True)
        self.reduced_train_df = self.train_merged_df[self.train_merged_df['Open'] == 1].copy()
        logging.info("Data preprocessed successfully.")
    
    def check_data_integrity(self):
        """Check for null values and data consistency."""
        logging.info("Checking data integrity...")
        logging.info(f"Null values:\n{self.train_merged_df.isnull().sum()}")
        logging.info(f"Unique value counts:\n{self.train_merged_df.nunique()}")
    
    def visualize_distribution(self):
        """Visualize sales and assortment distributions."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.reduced_train_df['Sales'], bins=50, kde=True)
        plt.title('Sales Distribution')
        plt.show()
        
        sns.histplot(self.reduced_train_df['Assortment'], discrete=True)
        plt.title('Assortment Distribution')
        plt.show()
    
    def visualize_relationships(self):
        """Visualize relationships between key variables."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Sales', y='Customers', data=self.reduced_train_df)
        plt.title('Sales vs Customers')
        plt.show()
        
        sns.barplot(x='StateHoliday', y='Sales', data=self.reduced_train_df)
        plt.title('Sales by StateHoliday')
        plt.show()
        
        sns.barplot(x='DayOfWeek', y='Sales', data=self.reduced_train_df)
        plt.title('Sales by Day of the Week')
        plt.show()
    
    def monthly_sales_trend(self):
        """Analyze monthly sales trends."""
        self.reduced_train_df['Date'] = pd.to_datetime(self.reduced_train_df['Date'])
        self.reduced_train_df['Year'] = self.reduced_train_df['Date'].dt.year
        self.reduced_train_df['Month'] = self.reduced_train_df['Date'].dt.month
        
        monthly_sales = self.reduced_train_df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
        monthly_sales['YearMonth'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str)
        
        plt.figure(figsize=(14, 7))
        sns.lineplot(x='YearMonth', y='Sales', data=monthly_sales, marker='o')
        plt.xticks(rotation=45)
        plt.title('Monthly Sales Trend')
        plt.tight_layout()
        plt.show()
    
    def run_pipeline(self):
        """Run the full EDA pipeline."""
        self.load_data()
        self.merge_data()
        self.preprocess_data()
        self.check_data_integrity()
        self.visualize_distribution()
        self.visualize_relationships()
        self.monthly_sales_trend()
        logging.info("EDA pipeline executed successfully.")
