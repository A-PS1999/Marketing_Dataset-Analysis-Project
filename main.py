import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

df = pd.read_csv('marketing_data.csv')

# remove whitespace in dataframe headers
df.columns = df.columns.str.strip()

# turn income column values into float, remove $ sign and comma. Needed for later processing.
df['Income'] = df['Income'].str.replace('$', '')
df['Income'] = df['Income'].str.replace(',', '')
df['Income'] = df['Income'].astype(float)

# below finds empty (NaN) cells in the dataframe which will need dealing with.
print(df.isnull().sum().sort_values(ascending=False))

# above found 24 NaN cells in 'Income' column. Below replaces these empty cells with median for Income.
df['Income'] = df['Income'].fillna(df['Income'].median())

# below shows that 'Dt_Customer' values are 'object' type, but would be better as 'datetime' format.
# subsequently converts them to datetime
print('\n', df['Dt_Customer'].dtype)
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

# create some new dataframe columns for potentially useful variables.
# total number of dependents (children, teenagers)
df['Total_Dependents'] = df['Teenhome'] + df['Kidhome']

# from Dt_Customer, which is exact day customer enrolled with company, get just the year
df['Customer_Join_Year'] = pd.DatetimeIndex(df['Dt_Customer']).year

# create column for total amount spent by customer
amount_columns = []

for col in df.columns:
    if 'Mnt' in col:
        amount_columns.append(col)

df['Total_Spend'] = df[amount_columns].sum(axis=1)

# create column for total num of purchases
purchase_columns = []

for col in df.columns:
    if 'Purchases' in col:
        purchase_columns.append(col)

df['Total_Purchases'] = df[purchase_columns].sum(axis=1)

# column for total number of campaigns accepted
campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']

df['Total_Accepted_Cmps'] = df[campaign_columns].sum(axis=1)

# below boxplot shows some outliers in customer years of birth.
# plot shows that some customers have DoB at or before 1900.
df.boxplot(
    column='Year_Birth'
)
plt.show()

# given above, below removes these outliers
df = df[df['Year_Birth'] > 1900].reset_index(drop=True)

# remove largely unnecessary 'ID' and 'Dt_Customer' columns
df.drop(columns=['ID', 'Dt_Customer'], inplace=True)

# path for saving below visualisations
savepath = 'C:/Users/Samuel/PycharmProjects/DatasetAnalysisProject/Visualisations/'

# below creates boxplot which displays relation between number of dependents and total spend.
# shows that, broadly speaking, total spend decreases as number of dependents increases.
df.boxplot(
    column='Total_Spend',
    by='Total_Dependents',
    figsize=(10, 7)
)
plt.title('Boxplots showing relation between number of dependents and total spend')
plt.suptitle("")
plt.savefig(savepath + 'dependents_and_spend_boxplot.png')
plt.show()

# below boxplot shows relation between number of dependents and number of deals purchased
df.boxplot(
    column='NumDealsPurchases',
    by='Total_Dependents',
    figsize=(10, 7)
)
plt.title('Boxplots showing relation between number of dependents and number of deals purchased')
plt.suptitle("")
plt.savefig(savepath + 'dependents_and_deals_boxplot.png')
plt.show()

# bar plot for total spend by country
df.groupby('Country')['Total_Spend'].sum().sort_values(ascending=False).plot(
    kind='bar',
    figsize=(10, 7),
    title='Total Spend by Country',
    ylabel='Amount Spent'
)
plt.savefig(savepath + 'total_spend_by_country_bar.png')
plt.show()

# bar plot for total purchases by country
df.groupby('Country')['Total_Purchases'].sum().sort_values(ascending=False).plot(
    kind='bar',
    figsize=(10, 7),
    title='Total Purchases by Country',
    ylabel='Amount Purchased'
)
plt.savefig(savepath + 'total_purchases_by_country_bar.png')
plt.show()

# code for creating choropleth map for success of marketing campaigns by country
# change some country codes from dataset so they work properly
df['Country_Codes'] = df['Country'].replace(
    {'CA': 'CAN', 'SA': 'ZAF', 'US': 'USA', 'ME': 'MEX', 'SP': 'ESP'}
)

campaigns_df = df[['Country_Codes', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
                   'AcceptedCmp5', 'Response']].melt(
    id_vars='Country_Codes', var_name='Campaign', value_name='% accepted'
)
campaigns_df = pd.DataFrame(
    campaigns_df.groupby(['Country_Codes', 'Campaign'])['% accepted'].mean()*100).reset_index(drop=False)

campaigns_map = px.choropleth(
    campaigns_df,
    locationmode='ISO-3',
    color='% accepted',
    facet_col='Campaign',
    facet_col_wrap=3,
    projection='natural earth',
    locations='Country_Codes',
    title='Marketing Campaign Percentage Success Rate by Country'
)
campaigns_map.write_image(
    savepath + 'marketing_campaign_success_by_country.png',
    width=700,
    height=500,
    scale=3
)
campaigns_map.show()

# plot illustrating relation between income and spending
fig = px.scatter(
    data_frame=df,
    x='Income',
    y='Total_Spend',
    range_x=[0, 200000],
    trendline='lowess',
    title='Scatter plot illustrating the relationship between income and total spend'
)
fig.write_image(
    savepath + 'relation_between_income_and_total_spend_scatter.png',
    scale=3
)
fig.show()

# create heatmap visualisation for dataframe to see correlations
# first is creation of correlation matrix. 'kendall' method used because some dataframe features are binary
corr_matrix = df.select_dtypes(include='number').corr(method='kendall')

heatmap = px.imshow(
    corr_matrix,
    color_continuous_scale=px.colors.sequential.RdBu,
    title='Heatmap of correlations in marketing_data.csv'
)
heatmap.write_image(
    savepath + 'marketing_data_correlations_heatmap.png',
    scale=3
)
heatmap.show()

# next, I begin taking steps necessary for constructing linear regression model in order to better
# predict features that influence purchases.
# below line isolates categorical features for encoding
categorical_feats = df.select_dtypes(exclude='number')

encoder = OneHotEncoder(sparse=False).fit(categorical_feats)

# turn encoded categorical features into dataframe
categoricals_encoded = pd.DataFrame(encoder.transform(categorical_feats))
categoricals_encoded.columns = encoder.get_feature_names(categorical_feats.columns)

# merge encoded categorical features with numeric ones
numericals = df.drop(columns=categorical_feats.columns)
df2 = pd.concat([categoricals_encoded, numericals], axis=1)

# below line isolates variables to test against each other, namely 'NumStorePurchases' as y vs other columns for x
X, y = df2.drop(columns='NumStorePurchases'), df2['NumStorePurchases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# linear regression model for aforementioned predictions
linreg = LinearRegression().fit(X_train, y_train)

predicts = linreg.predict(X_test)

# examine accuracy and quality of model and predictions through RMSE (root-mean-square error)
print("The linear regression model's RMSE is: {0}".format(mean_squared_error(y_test, predicts, squared=False)))

# calculate permutation importance to identify features that are most predictive of store purchases
results = permutation_importance(linreg, X_test, y_test)
importances = results.importances_mean

# create a pandas Series for the permutation importances so that they can be manipulated and show feature names
# more easily.
importances_series = pd.Series(importances, index=X_train.columns)

# create horizontal bar plot to illustrate top 5 most significant features
importances_series.nlargest(5).plot(kind='barh')
plt.title('Top 5 significant features for affecting number of store purchases')
plt.savefig(savepath + 'NumStorePurchases_permutation_feature_importance_bar.png',
            bbox_inches='tight',
            dpi=300
            )
plt.show()
