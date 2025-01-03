import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, data_paths):
        """
        Initialize with data file paths.
        Args:
            data_paths (dict): Dictionary containing paths to train, test, store, and sample submission files.
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
        """Load data from the specified file paths."""
        self.sample_submission = pd.read_csv(self.data_paths['sample_submission'])
        self.store = pd.read_csv(self.data_paths['store'])
        self.test = pd.read_csv(self.data_paths['test'])
        self.train = pd.read_csv(self.data_paths['train'])
        print("Data loaded successfully.")
    
    def merge_data(self):
        """Merge store data with train and test datasets."""
        self.train_merged_df = self.train.merge(self.store, how='left', on='Store').drop(['PromoInterval'], axis=1)
        self.test_merged_df = self.test.merge(self.store, how='left', on='Store')
        print("Data merged successfully.")
    
    def preprocess_data(self):
        """Handle null values and data inconsistencies."""
        self.train_merged_df['StateHoliday'].replace({'0': 0}, inplace=True)
        self.test_merged_df['StateHoliday'].replace({'0': 0}, inplace=True)
        self.reduced_train_df = self.train_merged_df[self.train_merged_df.Open == 1].copy()
        print("Data preprocessed successfully.")
    
    def check_data_integrity(self):
        """Check for null values and unique counts."""
        print("Null values in train dataset:\n", self.train_merged_df.isnull().sum())
        print("\nUnique values per column:\n", self.train_merged_df.nunique())
        print("\nStateHoliday value counts:\n", self.train_merged_df.StateHoliday.value_counts())
    
    def visualize_distribution(self):
        """Visualize the distribution of key columns."""
        sns.histplot(data=self.reduced_train_df, x='Sales')
        plt.title("Sales Distribution")
        plt.show()
        
        sns.histplot(data=self.reduced_train_df, x='Assortment')
        plt.title("Assortment Distribution")
        plt.show()
        
    def visualize_relationships(self):
        """Visualize relationships between key variables."""
        plt.figure(figsize=(20, 10))
        sns.scatterplot(x=self.reduced_train_df.Sales, y=self.reduced_train_df.Customers)
        plt.title("Sales vs Customers")
        plt.show()
        
        sns.barplot(data=self.reduced_train_df, x='StateHoliday', y='Sales')
        plt.title("Sales by StateHoliday")
        plt.show()
        
        sns.barplot(data=self.reduced_train_df, x='DayOfWeek', y='Sales')
        plt.title("Sales by Day of the Week")
        plt.show()
        
        sns.barplot(data=self.reduced_train_df, x='Assortment', y='Sales')
        plt.title("Sales by Assortment Type")
        plt.show()
        
        sns.barplot(data=self.reduced_train_df, x='StoreType', y='Sales')
        plt.title("Sales by Store Type")
        plt.show()
    
    def monthly_sales_trend(self):
        """Analyze and visualize monthly sales trends."""
        self.reduced_train_df['Date'] = pd.to_datetime(self.reduced_train_df['Date'])
        self.reduced_train_df['Year'] = self.reduced_train_df['Date'].dt.year
        self.reduced_train_df['Month'] = self.reduced_train_df['Date'].dt.month
        self.reduced_train_df['Day'] = self.reduced_train_df['Date'].dt.day
        
        monthly_sales = self.reduced_train_df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(
            monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str), 
            monthly_sales['Sales'], 
            marker='o'
        )
        plt.title('Monthly Sales Over Time')
        plt.xlabel('Year-Month')
        plt.ylabel('Sales')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def run_pipeline(self):
        """Run the entire EDA pipeline."""
        self.load_data()
        self.merge_data()
        self.preprocess_data()
        self.check_data_integrity()
        self.visualize_distribution()
        self.visualize_relationships()
        self.monthly_sales_trend()
