import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv(r"API_SE.XPD.TOTL.GD.ZS_DS2_en_csv_v2_247510.csv")

Non_countries = ['Africa Eastern and Southern', 'Africa Western and Central', 'Central Europe and the Baltics', 'Channel Islands', 'Caribbean small states', 'East Asia & Pacific (excluding high income)','Early-demographic dividend','East Asia & Pacific','Europe & Central Asia (excluding high income)','Europe & Central Asia', 'Euro area','European Union','Fragile and conflict affected situations','Micronesia, Fed. Sts.', 'High income',  'Heavily indebted poor countries (HIPC)','IBRD only',
 'IDA & IBRD total',
 'IDA total',
 'IDA blend',
'Isle of Man',
'Not classified',
'Latin America & Caribbean (excluding high income)',
'Latin America & Caribbean',
 'Least developed countries: UN classification',
 'Low income',
'Lower middle income',
 'Low & middle income',
'Late-demographic dividend',
'Middle East & North Africa',
'Middle income',
'Middle East & North Africa (excluding high income)',
'OECD members',
'Other small states',
'Pre-demographic dividend',
'Pacific island small states',
 'Post-demographic dividend',
 'Sub-Saharan Africa (excluding high income)',
'Sub-Saharan Africa',
 'Small states',
'East Asia & Pacific (IDA & IBRD countries)',
 'Europe & Central Asia (IDA & IBRD countries)',
'Latin America & the Caribbean (IDA & IBRD countries)',
'Middle East & North Africa (IDA & IBRD countries)',
'South Asia (IDA & IBRD)',
 'Sub-Saharan Africa (IDA & IBRD countries)',
'Upper middle income',
'IDA only',
'North America',
'World',]

Countries = [c for c in df['Country Name'].values if c not in Non_countries]

df.set_index('Country Name', inplace = True)
df.drop(columns = ['Country Code', 'Indicator Name', 'Indicator Code'], inplace = True)

df_clean_1 = df.loc[Countries]

countries_less_than_50_NaN = df_clean_1.isna().mean(axis = 1) < 0.5
years_less_than_50_NaN = df_clean_1.isna().mean(axis = 0) < 0.5

df_clean_2 = df_clean_1.loc[countries_less_than_50_NaN, years_less_than_50_NaN]

countries_with_zero_NaN = df_clean_2.isna().mean(axis = 1) == 0

df_clean_3 = df_clean_2.loc[countries_with_zero_NaN]

list_of_countries = (list(df_clean_3.index))
df_clean_3.columns = pd.period_range(start = '1999', end = '2021', freq = 'Y')

df_clean_4 = df_clean_3.stack().to_frame().rename(columns = {0:'percentage of GDP'})

df_clean_4 = df_clean_4.rename_axis(['Country Name', 'Years'])
df_clean_4.to_csv("df-clean-4.csv")

