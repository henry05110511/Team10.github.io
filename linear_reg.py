import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def identify_level(item_id, store, sales, year, target_sales_date):
    file_name = f"percentage_changes_decr_price/{year-1}_percentage_changes.csv"

    df = pd.read_csv(file_name)

    if not df[df['item_id'] == item_id].empty:
        if len(df[(df['item_id'] == item_id) & (df['store_id'] == store)]) < 20:
            if len(df[df['item_id'] == item_id]) < 20:
                dept_id = item_id[:-4]
                if len(df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]) < 20:
                    department_level_model(item_id, store, sales, year-1, target_sales_date, df)
                else:
                    department_store_level_model(item_id, store, sales, year-1, target_sales_date, df)
            else:
                item_level_model(item_id, store, sales, year-1, target_sales_date, df)
        else:
            item_store_level_model(item_id, store, sales, year-1, target_sales_date, df)
    else:
        print('Item not available for the year')

def item_level_model(item_id, store, sales, year, target_sales_date, df):
    df = df[df['item_id'] == item_id]
    df['price_change'] = df['price_change'].abs()
    fit_linear_model(df)

def item_store_level_model(item_id, store, sales, year, target_sales_date, df):
    df = df[(df['item_id'] == item_id) & (df['store_id'] == store)]
    df['price_change'] = df['price_change'].abs()
    fit_linear_model(df)

def department_level_model(item_id, store, sales, year, target_sales_date, df):
    dept_id = item_id[:-4]
    df = df[df['dept_id'] == dept_id]
    df['price_change'] = df['price_change'].abs()
    fit_linear_model(df)

def department_store_level_model(item_id, store, sales, year, target_sales_date, df):
    dept_id = item_id[:-4]
    df = df[(df['dept_id'] == dept_id) & (df['store_id'] == store)]
    df['price_change'] = df['price_change'].abs()
    fit_linear_model(df)


def fit_linear_model(df):
    df = df.dropna()

    # Prepare data
    X2 = df[['price_change']]  # Features (price change)
    #X = np.clip(X2, -100, 100)
    y2 = df['sales_change']     # Target variable (sales change)
    #y = np.clip(y2, -100, 100)

    # Select model
    model = LinearRegression()

    # Train model
    model.fit(X2, y2)

    # Predict sales change
    predicted_sales_change = model.predict(X2)

    #df['sales_change'] = np.clip(df['sales_change'], -100, 100)
    #df['price_change'] = np.clip(df['price_change'], -100, 100)
    # Add predicted sales change to DataFrame
    df['predicted_sales_change'] = predicted_sales_change


    # Plot actual vs predicted sales change
    plt.figure(figsize=(10, 6))
    plt.scatter(df['price_change'], df['sales_change'], color='blue', label='Actual Sales Change')
    plt.plot(df['price_change'], df['predicted_sales_change'], color='red', linewidth=2, label='Predicted Sales Change')
    plt.xlabel('Price Change (%)')
    plt.ylabel('Sales Change (%)')
    dep = df['dept_id'].unique()
    plt.title(f'Price Elasticity for {dep}')
    plt.legend()
    plt.grid(True)
    plt.show()


identify_level('FOODS_1_002', 'TX_1', 500000, 2012, '12/08/2012')