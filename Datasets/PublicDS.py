import pandas as pd
from Utils import Util
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.preprocessing import MinMaxScaler # For optional feature scaling


def get_ice_creame_sales_data_DS(path):
    """
    Loads the Ice Cream Sales dataset and returns features (X) and target (y).
    Source: https://www.kaggle.com/datasets/mirajdeepbhandari/polynomial-regression
    """
    data = pd.read_csv(path)
    X = data[["Temperature (°C)"]].values  # Ensure X is 2D
    y = data["Ice Cream Sales (units)"].values
    return X, y

def get_concrete_data_DS(path: object) -> object:
    """
    Abstract: Concrete is the most important material in civil engineering. The
    concrete compressive strength is a highly nonlinear function of age and
    ingredients. These ingredients include cement, blast furnace slag, fly ash,
    water, superplasticizer, coarse aggregate, and fine aggregate.
    Source: https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set
    """
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()  # Remove extra spaces from column names

    y = data['concrete_compressive_strength'].to_numpy()
    X = data.drop(columns=['concrete_compressive_strength']).to_numpy()

    # Standardize both
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    return X, y


def get_NASA_airfoil_self_noise_DS(path):
    """
    Loads the NASA Airfoil Self-Noise dataset and returns standardized X and y.

    Source: https://www.kaggle.com/datasets/fedesoriano/airfoil-selfnoise-dataset/code
    The last column is the target (Sound pressure level).
    """
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    # y = Last column (Sound pressure level)
    y = data.iloc[:, -1].values

    # X = All but last column
    X = data.iloc[:, :-1].values

    # Standardize both
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    return X, y


def get_parkinsons_telemonitoring_ucirvine_DS(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    # Set target variable
    y = data['total_updrs'].values

    # Drop non-feature columns
    X = data.drop(columns=['subject', 'test_time', 'motor_updrs', 'total_updrs']).values

    # Standardize
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    return X, y


def get_boston_house_price_DS(path):
    """
    Loads the Boston House Price dataset from a given CSV path,
    separates features (X) and the target (y), and standardizes both.

    The dataset is expected to have 'MEDV' as the target variable.
    Source: https://www.kaggle.com/datasets/arunjathari/bostonhousepricedata

    Args:
        path (str): The file path to the Boston House Price CSV dataset.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Standardized feature matrix.
            - y (numpy.ndarray): Standardized target vector.
    """
    # Load the dataset from the specified path
    data = pd.read_csv(path)
    # Strip any leading/trailing whitespace from column names for consistent access
    data.columns = data.columns.str.strip()

    # Separate features (X) and the target variable (y)
    # 'MEDV' is the median value of owner-occupied homes in $1000s, which is our target.
    # All other columns are considered features.
    X = data.drop('MEDV', axis=1) # Drop the 'MEDV' column to get features
    y = data['MEDV']             # Select the 'MEDV' column as the target

    # Standardize the features (X)
    # StandardScaler transforms data to have a mean of 0 and a standard deviation of 1.
    # This is crucial for many machine learning algorithms.
    X = StandardScaler().fit_transform(X)

    # Standardize the target (y)
    # The target 'y' is a 1D Series, but StandardScaler expects a 2D array.
    # We reshape it to (-1, 1) to make it a column vector, then standardize,
    # and finally use .ravel() to convert it back to a 1D array.
    y = StandardScaler().fit_transform(y.values.reshape(-1, 1)).ravel()

    return X, y

def get_health_DS(path):
    """
    Loads the Health and Lifestyle dataset from a given CSV path,
    preprocesses it (one-hot encodes categorical features, separates X and y),
    and standardizes both features (X) and the target (y).

    The dataset is expected to contain columns related to health and lifestyle,
    with 'Health_Score' being used as the default target variable (y).
    Source: https://www.kaggle.com/datasets/pratikyuvrajchougule/health-and-lifestyle-data-for-regression

    Args:
        path (str): The file path to the Health and Lifestyle CSV dataset.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Standardized feature matrix.
            - y (numpy.ndarray): Standardized target vector.
    """
    # Load the dataset from the specified path
    data = pd.read_csv(path)
    # Strip any leading/trailing whitespace from column names for consistent access
    data.columns = data.columns.str.strip()

    # Define the target variable (y)
    # Based on the provided columns, 'Health_Score' is a suitable numerical target for regression.
    target_column = 'Health_Score'
    y = data[target_column]

    # Define features (X) by dropping the target and other non-feature columns.
    # 'Person ID' and 'Blood Pressure (systolic/diastolic)' are not present in this dataset.
    features_to_drop = [target_column] # Only dropping the target column from features
    X = data.drop(columns=features_to_drop, errors='ignore')

    # Identify categorical and numerical columns for preprocessing
    # Based on the provided columns: 'Smoking_Status' and 'Alcohol_Consumption' are likely categorical.
    categorical_features = ['Smoking_Status', 'Alcohol_Consumption']
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Create a preprocessor using ColumnTransformer for different transformations
    # OneHotEncoder for categorical features and StandardScaler for numerical features.
    # Set sparse_output=False in OneHotEncoder to ensure a dense array output,
    # which resolves the TypeError: sparse array length is ambiguous.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # Apply preprocessing to features (X)
    X_processed = preprocessor.fit_transform(X)

    # Standardize the target (y)
    # StandardScaler expects a 2D array, so reshape y.
    # .ravel() converts it back to a 1D array after standardization.
    y_standardized = StandardScaler().fit_transform(y.values.reshape(-1, 1)).ravel()

    return X_processed, y_standardized



def get_forest_fire_DS(path):
    """
    Loads the Forest Fires Data Set from a given path, separates it into
    features (X) and target (y), preprocesses categorical features
    using one-hot encoding, and converts the output to NumPy arrays.

    Args:
        path (str): The file path to the forestfires.csv dataset.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The preprocessed features of the dataset as a NumPy array.
            - y (np.ndarray): The target variable (burnt area) as a NumPy array.
            - None, None if there's a FileNotFoundError or other loading error.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found. Please check the path.")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return None, None

    # 'area' is the target variable (burnt area in hectares)
    y = df['area']

    # X will be all other columns (features)
    X = df.drop('area', axis=1)

    # Identify categorical columns for one-hot encoding
    categorical_cols = ['month', 'day']

    # Apply one-hot encoding to the categorical columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Ensure all remaining columns in X are numeric.
    for col in X.columns:
        if X[col].dtype == 'object':
            print(
                f"Warning: Column '{col}' is still of object type after one-hot encoding. Attempting to convert to numeric.")
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if X[col].isnull().any():
                print(
                    f"Warning: Column '{col}' contains non-numeric values that were converted to NaN. Consider handling these NaNs (e.g., fill or drop).")

    # Crucial change: Convert DataFrames/Series to NumPy arrays before returning
    # This ensures your kernel functions receive NumPy arrays which have the .reshape() method.
    X_np = X.values.astype(float)
    y_np = y.values.astype(float)
    y_np = np.log1p(y.values).astype(float)

    return X_np, y_np


def get_wine_quality_DS(red_wine_path, white_wine_path, wine_type_filter='both', scale_features=False, normalize_target=False):
    """
    Loads and preprocesses the Wine Quality dataset.
    Allows filtering for red wine, white wine, or both.
    Combines data, adds a 'wine_type' feature, separates X and y,
    and converts them to NumPy arrays.

    Args:
        red_wine_path (str): The file path to 'winequality-red.csv'.
        white_wine_path (str): The file path to 'winequality-white.csv'.
        wine_type_filter (str): Specifies which wine data to load.
                                 'red': Loads only red wine data.
                                 'white': Loads only white wine data.
                                 'both': Loads both red and white wine data (default).
        scale_features (bool): If True, applies MinMaxScaler to the features (X).
        normalize_target (bool): If True, applies min-max scaling to the target (y)
                                 to scale it between 0 and 1.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The preprocessed features as a NumPy array.
            - y (np.ndarray): The target variable (quality) as a NumPy array.
            - None, None if there's a FileNotFoundError or other loading error.
    """
    df_list = []

    try:
        if wine_type_filter in ['red', 'both']:
            red_df = pd.read_csv(red_wine_path, sep=';')
            red_df['wine_type'] = 0  # 0 for red wine
            df_list.append(red_df)
            print(f"Loaded red wine data: {len(red_df)} samples")

        if wine_type_filter in ['white', 'both']:
            white_df = pd.read_csv(white_wine_path, sep=';')
            white_df['wine_type'] = 1 # 1 for white wine
            df_list.append(white_df)
            print(f"Loaded white wine data: {len(white_df)} samples")

        if not df_list: # Check if no data was loaded due to invalid filter
            print(f"Error: Invalid 'wine_type_filter' specified: '{wine_type_filter}'. Must be 'red', 'white', or 'both'.")
            return None, None

        # Combine the datasets
        df = pd.concat(df_list, ignore_index=True)
        print(f"Combined dataset: {len(df)} samples")

    except FileNotFoundError as e:
        print(f"Error: One of the files was not found: {e}. Please check the paths.")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading or combining CSVs: {e}")
        return None, None

    # Separate features (X) and target (y)
    y = df['quality']
    X = df.drop('quality', axis=1)

    # Convert features to NumPy array
    X_np = X.values.astype(float)

    # Convert target to NumPy array
    y_np = y.values.astype(float)

    # Optional: Scale features
    if scale_features:
        scaler_X = MinMaxScaler()
        X_np = scaler_X.fit_transform(X_np)
        print("Features scaled using MinMaxScaler.")

    # Optional: Normalize target (quality)
    if normalize_target:
        y_np_reshaped = y_np.reshape(-1, 1)
        scaler_y = MinMaxScaler()
        y_np = scaler_y.fit_transform(y_np_reshaped).flatten()
        print("Target 'quality' normalized using MinMaxScaler.")

    print("Dataset loaded and preprocessed successfully.")
    return X_np, y_np


def get_medical_cost_insurance_DS(path, scale_features=True, transform_target=True):
    """
    github: https://www.kaggle.com/datasets/mirichoi0218/insurance
    Loads and preprocesses the Medical Cost Personal Insurance Plan dataset.
    Handles categorical features, separates X and y, and optionally scales features
    and transforms the target variable.

    Args:
        path (str): The file path to 'insurance.csv'.
        scale_features (bool): If True, applies MinMaxScaler to the features (X).
                               Defaults to True.
        transform_target (bool): If True, applies np.log1p to the 'charges' column (y).
                                 Defaults to True.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The preprocessed features as a NumPy array.
            - y (np.ndarray): The target variable (charges) as a NumPy array.
                              Transformed if transform_target is True.
            - None, None if there's a FileNotFoundError or other loading error.
    """
    try:
        df = pd.read_csv(path)
        print(f"Loaded dataset: {len(df)} samples")
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found. Please check the path.")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return None, None

    # Identify categorical columns for one-hot encoding
    # Based on dataset description: 'sex', 'smoker', 'region' are categorical
    categorical_cols = ['sex', 'smoker', 'region']

    # Apply one-hot encoding to the categorical columns
    # drop_first=True helps prevent multicollinearity
    X = pd.get_dummies(df.drop('charges', axis=1), columns=categorical_cols, drop_first=True)

    # The target variable is 'charges'
    y = df['charges']

    # Convert features to NumPy array and ensure float type
    X_np = X.values.astype(float)

    # Apply transformation to y if requested (log(x+1) for skewed cost data)
    if transform_target:
        y_np = np.log1p(y.values).astype(float)
        print("Target 'charges' transformed using np.log1p.")
    else:
        y_np = y.values.astype(float)

    # Optional: Scale features
    if scale_features:
        scaler_X = MinMaxScaler()
        X_np = scaler_X.fit_transform(X_np)
        print("Features scaled using MinMaxScaler.")

    print("Dataset loaded and preprocessed successfully.")
    return X_np, y_np

if __name__ == "__main__":
    # path = Util.get_dataset_path('Ice_cream_selling_data.csv')
    # X, y = get_ice_creame_sales_data_DS(path)
    # print(X)
    # print(y)
    # print(X.shape)

    #########################################################
    # path = Util.get_dataset_path('concrete_data.csv')
    # X, y = get_concrete_data_DS(path)
    ##########################################################
    # path = Util.get_dataset_path('KAG_energydata_complete.csv')
    # X, y = get_appliances_energy_prediction_DS(path)
    ##########################################################
    # path = Util.get_dataset_path('KAG_energydata_complete.csv')
    # X, y = get_appliances_energy_prediction_DS2(path, corr_threshold=0.3, min_features=5)
    # print(X.shape)  # e.g., (19735, 5) if only 5 features are strong
    ##########################################################
    # path = Util.get_dataset_path('AirfoilSelfNoise.csv')
    # X, y = get_NASA_airfoil_self_noise_DS(path)
    # print(X.shape)
    #########################################################
    # path = Util.get_dataset_path("Parkinsons-Telemonitoring-ucirvine.csv")
    # X, y = get_parkinsons_telemonitoring_ucirvine_DS(path)
    # print(X.shape)
    #############################################################
    # path = Util.get_dataset_path("Boston-house-price-data.csv")
    # X, y = get_boston_house_price_DS(path)
    # print(X.shape)
    #############################################################
    # path = Util.get_dataset_path("health_data.csv")
    # X, y = get_health_DS(path)
    # print(X.shape)
    #############################################################
    # path = Util.get_dataset_path("forestfires.csv")
    # X,y = get_forest_fire_DS(path)
    # print(X.shape)
    ############################################################
    # path1 = Util.get_dataset_path("winequality-red.csv")
    # path2 = Util.get_dataset_path("winequality-white.csv")
    # X, y = get_wine_quality_DS(path1, path2, "red", True, True)
    # print(X.shape)
    ###########################################################
    path = Util.get_dataset_path("insurance.csv")
    X, y = get_medical_cost_insurance_DS(path)
    print(X.shape)





