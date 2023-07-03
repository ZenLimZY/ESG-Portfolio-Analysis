#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.api as sm
import numpy as np

# User Input number of trading days and target return
Trading_Days = 252
target_return = 0
xlsx_file = "Assignment ESG.xlsx"

#---Qn0 Data preparation, to split all sheets in the excel file

# Read the Excel file
xls = pd.ExcelFile(xlsx_file)

# Iterate through each sheet in the XLSX file
for sheet_name in xls.sheet_names:
    # Read the sheet into a DataFrame
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # Create a CSV file name based on the sheet name
    csv_file = f"{sheet_name}.csv"
    
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)
    print(f"Sheet '{sheet_name}' is saved as '{csv_file}'")

# Read the CSV file
file_path = 'GMB.csv'
df = pd.read_csv(file_path)

# Rename columns B, C, and D
df.rename(columns={'B': 'ret_brown', 'C': 'ret_green', 'D': 'ret_gmb'}, inplace=True)

# Calculate cumulative returns for each column
cumulative_returns_brown = (1 + df['ret_brown'].pct_change()).cumprod() - 1
cumulative_returns_green = (1 + df['ret_green'].pct_change()).cumprod() - 1
cumulative_returns_gmb = (1 + df['ret_gmb'].pct_change()).cumprod() - 1

#--- Qn1 - Calculate cumulative returns for portfolios and GMB factor 

# Add cumulative returns to the DataFrame
df['Cumulative Returns Brown'] = cumulative_returns_brown
df['Cumulative Returns Green'] = cumulative_returns_green
df['Cumulative Returns GMB'] = cumulative_returns_gmb

# Write DataFrame back to CSV file
df.to_csv(file_path, index=False)

print("Cumulative returns written to the CSV file successfully.")

#---- Qn 2 Calculate risk-adjusted returns with respect to different asset pricing models (FF5)

# Convert the 'DATE' column to datetime format
df['DATE'] = pd.to_datetime(df['DATE'])

# Read the CSV file for FF5
file_path = 'FF5.csv'
dfFF = pd.read_csv(file_path)

# Convert the 'DATE' column to datetime format
dfFF['DATE'] = pd.to_datetime(dfFF['DATE'])

# Merge the dataframes
merged_df = pd.merge(df, dfFF, on='DATE')

# Calculates Excess Returns on Brown
merged_df['excess_brown'] = merged_df['ret_brown'] - merged_df['RF']

# Calculates Excess Returns on green
merged_df['excess_green'] = merged_df['ret_green'] - merged_df['RF']

# Calculates Excess Returns on gmb
merged_df['excess_gmb'] = merged_df['ret_gmb'] - merged_df['RF']

# Convert the DataFrame to CSV
csv_file = 'MergedDF Qn1.5.csv'
merged_df.to_csv(csv_file, index=False)

# Calculate the annualized standard deviations (volatility) of the excess returns
volatility_brown = merged_df['excess_brown'].std() * np.sqrt(Trading_Days)
volatility_green = merged_df['excess_green'].std() * np.sqrt(Trading_Days)
volatility_gmb = merged_df['excess_gmb'].std() * np.sqrt(Trading_Days)

# Calculate the annualized mean of the excess returns
mean_excess_return_brown = merged_df['excess_brown'].mean() * Trading_Days
mean_excess_return_green = merged_df['excess_green'].mean() * Trading_Days
mean_excess_return_gmb = merged_df['excess_gmb'].mean() * Trading_Days

# Calculate the Sharpe Ratio for each security
sharpe_ratio_brown = mean_excess_return_brown / volatility_brown
sharpe_ratio_green = mean_excess_return_green / volatility_green
sharpe_ratio_gmb = mean_excess_return_gmb / volatility_gmb

# Calculate the downside deviation for each security
downside_deviation_brown = np.sqrt((merged_df['excess_brown'][merged_df['excess_brown'] < target_return]**2).mean()) * np.sqrt(Trading_Days)
downside_deviation_green = np.sqrt((merged_df['excess_green'][merged_df['excess_green'] < target_return]**2).mean()) * np.sqrt(Trading_Days)
downside_deviation_gmb = np.sqrt((merged_df['excess_gmb'][merged_df['excess_gmb'] < target_return]**2).mean()) * np.sqrt(Trading_Days)

# Calculate the Sortino Ratio for each security
sortino_ratio_brown = mean_excess_return_brown / downside_deviation_brown
sortino_ratio_green = mean_excess_return_green / downside_deviation_green
sortino_ratio_gmb = mean_excess_return_gmb / downside_deviation_gmb

# Run the linear regression to find beta for each security
beta_brown = sm.OLS(merged_df['excess_brown'], sm.add_constant(merged_df['MKTRF'])).fit().params['MKTRF']
beta_green = sm.OLS(merged_df['excess_green'], sm.add_constant(merged_df['MKTRF'])).fit().params['MKTRF']
beta_gmb = sm.OLS(merged_df['excess_gmb'], sm.add_constant(merged_df['MKTRF'])).fit().params['MKTRF']

# Calculate the Treynor Ratio for each security
treynor_ratio_brown = (mean_excess_return_brown - merged_df['RF'].mean() * Trading_Days) / beta_brown
treynor_ratio_green = (mean_excess_return_green - merged_df['RF'].mean() * Trading_Days) / beta_green
treynor_ratio_gmb = (mean_excess_return_gmb - merged_df['RF'].mean() * Trading_Days) / beta_gmb

# Create a new DataFrame with these values
df_output = pd.DataFrame({
    'Portfolio': ['brown', 'green', 'gmb'],
    'Volatility': [volatility_brown, volatility_green, volatility_gmb],
    'Mean Excess Return': [mean_excess_return_brown, mean_excess_return_green, mean_excess_return_gmb],
    'Sharpe Ratio': [sharpe_ratio_brown, sharpe_ratio_green, sharpe_ratio_gmb],
    'Downside Deviation': [downside_deviation_brown, downside_deviation_green, downside_deviation_gmb],
    'Sortino Ratio': [sortino_ratio_brown, sortino_ratio_green, sortino_ratio_gmb],
    'Beta': [beta_brown, beta_green, beta_gmb],
    'Treynor Ratio': [treynor_ratio_brown, treynor_ratio_green, treynor_ratio_gmb]
})

# Export the DataFrame to a CSV file
df_output.to_csv('Risk Adjusted Returns Q2.csv', index=False)
print("Risk Adjusted returns written to the CSV file successfully.")

#---Qn3 Calculate correlations between portfolio returns/GMB factor and MCCC index

# Read the CSV file for GMB
file_path = 'GMB.csv'
dfGMB = pd.read_csv(file_path)

# Trim dfGMB back to the original columns
columns_to_keep = ["DATE", "ret_brown", "ret_green", "ret_gmb", "gmb_factor"]
dfGMB = dfGMB[columns_to_keep]

# Read the CSV file for MCCC
file_path = 'MCCC.csv'
dfMCCC = pd.read_csv(file_path)

# Convert date column to datetime format
dfMCCC['date'] = pd.to_datetime(dfMCCC['Date'])
dfGMB['DATE'] = pd.to_datetime(dfGMB['DATE'])

# Roll back the date by one day (Need to roll back due to not alligning with GMB sheet)
dfMCCC['date'] -= pd.DateOffset(days=1)
# Rename the column to "DATE"
dfMCCC.rename(columns={'date': 'DATE'}, inplace=True)
# Rename the column aggregate  to "MCCC"
dfMCCC.rename(columns={'Aggregate': 'MCCC_Index'}, inplace=True)
# Drop the "Date" column
dfMCCC.drop('Date', axis=1, inplace=True)

# Merge the dataframes
merged_df = pd.merge(dfGMB, dfMCCC, on='DATE')
corr_matrix = merged_df.corr()
# Export the corr_matrix DataFrame to a CSV file
corr_matrix.to_csv('Correlation Matrix Q3.csv', index=False)
print("corr_matrix to the CSV file successfully.")

#---Qn4


# In[ ]:




