import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as ms


'''Data Preprocessing'''

files = [
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
]

data = pd.concat([pd.read_csv(file) for file in files])

rows,cols = data.shape
print(f"Number of rows: {rows}") #703245
print(f"Number of columns: {cols}") #79


col_names = {col: col.strip() for col in data.columns}
data.rename(columns = col_names, inplace = True)


'''Data Cleaning'''

data.drop_duplicates(inplace = True)

missing_val = data.isna().sum()
print(missing_val.loc[missing_val > 0])

numeric_cols = data.select_dtypes(include = np.number).columns
inf_count = np.isinf(data[numeric_cols]).sum()
#print(inf_count[inf_count > 0]) #Counting Infinity Values

data.replace([np.inf, -np.inf], np.nan, inplace = True)
data.dropna(inplace = True) #Replacing infinity values with NaN and dropping rows with NaN values

missing = data.isna().sum()
#print(missing.loc[missing > 0])

mis_per = (missing / len(data)) * 100
mis_table = pd.concat([missing, mis_per.round(2)], axis = 1)
mis_table = mis_table.rename(columns = {0 : 'Missing Values', 1 : 'Percentage of Total Values'})

#print(mis_table.loc[mis_per > 0]) #No missing values after dropna

#print("Shape after dropna:", data.shape) #(615433,79)

#plotting boxplot for 'Flow Packets/s' column to check for outliers and skewness
#plt.figure(figsize = (8, 3))
#sns.boxplot(x = data['Flow Packets/s'])
#plt.xlabel('Boxplot of Flow Packets/s')
#plt.show()

med_flow_bytes = data['Flow Bytes/s'].median()
med_flow_packets = data['Flow Packets/s'].median()

#print('Median of Flow Bytes/s: ', med_flow_bytes) #7039.473684
#print('Median of Flow Packets/s: ', med_flow_packets) #71.18451025

#print('Unique values in Label column:', data['Label'].unique()) #['BENIGN' 'Bot' 'PortScan' 'DDoS'] Note: This is not the final feature, I'm just assigning numerical values to the labels for correlation analysis. The actual model training will probably be done using the original categorical labels.
label_map = {
    'BENIGN': 0,
    'Bot': 1,
    'PortScan': 2,
    'DDoS': 3
}

data['Label_encoded'] = data['Label'].map(label_map)

fig, ax = plt.subplots(figsize = (24, 24))

#Using heatmap to se correlations between variables. Correlation values are rounded to 2 decimal places for better readability. The heatmap is colored using the 'coolwarm' colormap, which helps to visually distinguish between positive and negative correlations. The title of the heatmap is set to 'Correlation Matrix' with a font size of 18 for better visibility.
corr = data.corr(numeric_only = True).round(2)

#corr.style.background_gradient(cmap = 'coolwarm', axis = None).format(precision = 2)

#sns.heatmap(corr, cmap = 'coolwarm', annot = False, linewidth = 0.5)

#plt.xticks(fontsize=6, rotation=90)
#plt.yticks(fontsize=6)

#plt.title('Correlation Matrix', fontsize = 18)
#plt.show()

# Positive correlation features for 'Attack Number'
pos_corr_features = corr['Label_encoded'][(corr['Label_encoded'] > 0) & (corr['Label_encoded'] < 1)].index.tolist()

#Identifying features that have a positive correlation with 'Attack Number' and printing them in a formatted manner. The correlation values are also displayed alongside the feature names for better understanding of the strength of the correlation.
print("Features with positive correlation with 'Attack Number':\n")
for i, feature in enumerate(pos_corr_features, start = 1):
    corr_value = corr.loc[feature, 'Label_encoded']
    print('{:<3} {:<24} :{}'.format(f'{i}.', feature, corr_value))  #24 Features to consider

# Check class distribution For example: There might be more rows of a DDoS attack than a PortScan attack so we need to account for this imbalance
label_counts = data['Label'].value_counts()
print(label_counts)


