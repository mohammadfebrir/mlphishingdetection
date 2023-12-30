# -*- coding: utf-8 -*-
## Helper
"""

# import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# pandas config
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# common
def detect_outliers(df, features, threshold=1.5):
    outlier_indices = []

    for feature in features:
        Q1 = np.percentile(df[feature], 25)
        Q3 = np.percentile(df[feature], 75)

        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = np.where((df[feature] < lower_bound) | (df[feature] > upper_bound))[0]
        outlier_indices.extend(outliers)

    outlier_indices = list(set(outlier_indices))

    return outlier_indices

"""## Data Understanding"""

phising_df = pd.read_csv('Phising_Dataset.csv')
phising_df.head(20)

"""### Data Gathering

#### Data Shape

- `Phising_Dataset` terdiri dari 88647 records dan 112 fitur
"""

# data shape
phising_df.shape

"""#### Data Descriptions

##### Feature Explanation

1. `qty_dot_url` - The number of dots in the URL.
2. `qty_hyphen_url` - The number of hyphens in the URL.
3. `qty_underline_url` - The number of underscores in the URL.
4. `qty_slash_url` - The number of slashes in the URL.
5. `qty_questionmark_url` - The number of question marks in the URL.
6. `qty_equal_url` - The number of equal signs in the URL.
7. `qty_at_url` - The number of at symbols in the URL.
8. `qty_and_url` - The number of ampersands in the URL.
9. `qty_exclamation_url` - The number of exclamation marks in the URL.
10. `qty_space_url` - The number of spaces in the URL.
11. `qty_tilde_url` - The number of tilde characters in the URL.
12. `qty_comma_url` - The number of commas in the URL.
13. `qty_plus_url` - The number of plus signs in the URL.
14. `qty_asterisk_url` - The number of asterisks in the URL.
15. `qty_hashtag_url` - The number of hashtags in the URL.
16. `qty_dollar_url` - The number of dollar signs in the URL.
17. `qty_percent_url` - The number of percent signs in the URL.
18. `qty_tld_url` - The number of top-level domains in the URL.
19. `length_url` - The length of the URL.
20. `qty_dot_domain` - The number of dots in the domain.
21. `qty_hyphen_domain` - The number of hyphens in the domain.
22. `qty_underline_domain` - The number of underscores in the domain.
23. `qty_slash_domain` - The number of slashes in the domain.
24. `qty_questionmark_domain` - The number of question marks in the domain.
25. `qty_equal_domain` - The number of equal signs in the domain.
26. `qty_at_domain` - The number of at symbols in the domain.
27. `qty_and_domain` - The number of ampersands in the domain.
28. `qty_exclamation_domain` - The number of exclamation marks in the domain.
29. `qty_space_domain` - The number of spaces in the domain.
30. `qty_tilde_domain` - The number of tilde characters in the domain.
31. `qty_comma_domain` - The number of commas in the domain.
32. `qty_plus_domain` - The number of plus signs in the domain.
33. `qty_asterisk_domain` - The number of asterisks in the domain.
34. `qty_hashtag_domain` - The number of hashtags in the domain.
35. `qty_dollar_domain` - The number of dollar signs in the domain.
36. `qty_percent_domain` - The number of percent signs in the domain.
37. `qty_vowels_domain` - The number of vowels in the domain.
38. `domain_length` - The length of the domain.
39. `domain_in_ip` - Whether the domain is present in the IP address.
40. `server_client_domain` - The relationship between server and client in the domain.
41. `qty_dot_directory` - The number of dots in the directory.
42. `qty_hyphen_directory` - The number of hyphens in the directory.
43. `qty_underline_directory` - The number of underscores in the directory.
44. `qty_slash_directory` - The number of slashes in the directory.
45. `qty_questionmark_directory` - The number of question marks in the directory.
46. `qty_equal_directory` - The number of equal signs in the directory.
47. `qty_at_directory` - The number of at symbols in the directory.
48. `qty_and_directory` - The number of ampersands in the directory.
49. `qty_exclamation_directory` - The number of exclamation marks in the directory.
50. `qty_space_directory` - The number of spaces in the directory.
51. `qty_tilde_directory` - The number of tilde characters in the directory.
52. `qty_comma_directory` - The number of commas in the directory.
53. `qty_plus_directory` - The number of plus signs in the directory.
54. `qty_asterisk_directory` - The number of asterisks in the directory.
55. `qty_hashtag_directory` - The number of hashtags in the directory.
56. `qty_dollar_directory` - The number of dollar signs in the directory.
57. `qty_percent_directory` - The number of percent signs in the directory.
58. `directory_length` - The length of the directory.
59. `qty_dot_file` - The number of dots in the file.
60. `qty_hyphen_file` - The number of hyphens in the file.
61. `qty_underline_file` - The number of underscores in the file.
62. `qty_slash_file` - The number of slashes in the file.
63. `qty_questionmark_file` - The number of question marks in the file.
64. `qty_equal_file` - The number of equal signs in the file.
65. `qty_at_file` - The number of at symbols in the file.
66. `qty_and_file` - The number of ampersands in the file.
67. `qty_exclamation_file` - The number of exclamation marks in the file.
68. `qty_space_file` - The number of spaces in the file.
69. `qty_tilde_file` - The number of tilde characters in the file.
70. `qty_comma_file` - The number of commas in the file.
71. `qty_plus_file` - The number of plus signs in the file.
72. `qty_asterisk_file` - The number of asterisks in the file.
73. `qty_hashtag_file` - The number of hashtags in the file.
74. `qty_dollar_file` - The number of dollar signs in the file.
75. `qty_percent_file` - The number of percent signs in the file.
76. `file_length` - The length of the file.
77. `qty_dot_params` - The number of dots in the parameters.
78. `qty_hyphen_params` - The number of hyphens in the parameters.
79. `qty_underline_params` - The number of underscores in the parameters.
80. `qty_slash_params` - The number of slashes in the parameters.
81. `qty_questionmark_params` - The number of question marks in the parameters.
82. `qty_equal_params` - The number of equal signs in the parameters.
83. `qty_at_params` - The number of at symbols in the parameters.
84. `qty_and_params` - The number of ampersands in the parameters.
85. `qty_exclamation_params` - The number of exclamation marks in the parameters.
86. `qty_space_params` - The number of spaces in the parameters.
87. `qty_tilde_params` - The number of tilde characters in the parameters.
88. `qty_comma_params` - The number of commas in the parameters.
89. `qty_plus_params` - The number of plus signs in the parameters.
90. `qty_asterisk_params` - The number of asterisks in the parameters.
91. `qty_hashtag_params` - The number of hashtags in the parameters.
92. `qty_dollar_params` - The number of dollar signs in the parameters.
93. `qty_percent_params` - The number of percent signs in the parameters.
94. `params_length` - The length of the parameters.
95. `tld_present_params` - Whether the top-level domain is present in the parameters.
96. `qty_params` - The number of parameters.
97. `email_in_url` - Whether there is an email address in the URL.
98. `time_response` - The response time of the website.
99. `domain_spf` - Whether the domain uses SPF (Sender Policy Framework).
100. `asn_ip` - The Autonomous System Number (ASN) of the IP address.
101. `time_domain_activation` - The time of domain activation.
102. `time_domain_expiration` - The time of domain expiration.
103. `qty_ip_resolved` - The number of IP addresses resolved.
104. `qty_nameservers` - The number of nameservers for the domain.
105. `qty_mx_servers` - The number of mail servers for the domain.
106. `ttl_hostname` - The time-to-live value for the hostname.
107. `tls_ssl_certificate` - The type of SSL/TLS certificate used.
108. `qty_redirects` - The number of redirects.
109. `url_google_index` - Whether the URL is indexed by Google.
110. `domain_google_index` - Whether the domain is indexed by Google.
111. `url_shortened` - Whether the URL is shortened.
112. `phishing` - The target variable indicating whether the instance is phishing or not.
"""

for i in range (len(phising_df.columns)):
  print('{}. {}'.format(i+1, phising_df.columns[i]))

"""### Review Data

#### Attribute Characterisctics

##### Summary Statistics
"""

# data basic statistics
phising_df.describe()

"""##### Summary Types

Pada bagian ini dapat dilihat bahwa terdapat invalid type pada fitur qtt_dot_url yang seharusnya `float64` menjadi `object`.

Hal ini menandakan bahwa terdapat `noise` pada fitur tersebut.
"""

# data describe by feature types
phising_df.info(verbose=True)

"""##### Error Types"""

# feature that has too much
feature_error_type = []
for column in phising_df.columns:
  if(phising_df[column].mode().iloc[0] < 0):
    feature_error_type.append(column)

print(len(feature_error_type))
print(feature_error_type)

"""##### Missing Value

Dari missing value checking ini dapat diketahui bahwa terdapat missing value di 111 data lainnya, dengan maximum missing value count pada fitur `qty_asterisk_domain` sebanyak 4 missing values
"""

# checking missing value on every features
phising_df.isna().sum()

# maximimum missing value count
print(phising_df.isna().sum().idxmax(), phising_df.isna().sum().max())

# minimum missing value count
print(phising_df.isna().sum().idxmin(), phising_df.isna().sum().min())

"""##### Cardinality

Dari bagian ini dapat kita lihat dengan jelas
- Keberagaman data pada fitur
- Fitur dengan data yang salah sehingga membantu dalam `feature selection`

Data categorical
```
domain_in_ip, tld_present_params, server_client_domain, email_in_url, domain_spf, tls_ssl_certificate, url_shortened, phishing
```

Fitur dengan data yang rusak
```
qty_dot_url
```

Fitur yang dapat dihilangin
```
qty_slash_domain, qty_questionmark_domain, qty_equal_domain, qty_and_domain, qty_exclamation_domain, qty_space_domain, qty_tilde_domain, qty_comma_domain, qty_plus_domain, qty_asterisk_domain, qty_hashtag_domain, qty_dollar_domain, qty_percent_domain, qty_questionmark_directory, qty_hashtag_directory, qty_slash_file, qty_questionmark_file, qty_hashtag_file, qty_dollar_file, qty_hashtag_params
```
"""

for column in phising_df.columns:
  print('\n{}: {}'.format(column, phising_df[column].unique()))

"""##### Noise

###### Features

NB: First column skip because from the cardinality we know that it has error type on it
"""

nan_percentage = phising_df.isna().mean() * 100

negative_percentage = (phising_df.iloc[:, 1:] < 0).mean() * 100

columns_with_high_nan = nan_percentage[nan_percentage > 50].index
columns_with_high_negative = negative_percentage[negative_percentage > 50].index

columns_to_drop = set(columns_with_high_nan) | set(columns_with_high_negative)

print("This is column that could be dropped:", columns_to_drop)
print(len(columns_to_drop))

"""###### Records

NB: Remove first feature because inconsistent type for negative rows, It will be identified by nan_rows
"""

# threshold
threshold = 0.25 * len(phising_df.columns)
print(threshold)

# non float value on feature qty_for_url

non_float_records_index = []

for i, value in phising_df['qty_dot_url'].items():
  try:
    float_value = float(value)
  except (ValueError, TypeError):
    non_float_records_index.append(i)

    float_value = phising_df['qty_dot_url'][i]

  phising_df.at[i, 'qty_dot_url'] = float_value

non_float_records = phising_df.loc[non_float_records_index]
non_float_records

# Identify negative columns that more than threshold set
selected_columns = phising_df.columns[1:]

negative_condition = (phising_df[selected_columns] < 0).sum(axis=1) > threshold
records_with_negative_values = phising_df[negative_condition]
print(records_with_negative_values.shape)
records_with_negative_values.head(10)

# identify nan value on a record more than threshold
nan_condition = phising_df.isna().sum(axis=1) > threshold
records_with_nan_values = phising_df[nan_condition]
print(records_with_nan_values.shape)
records_with_nan_values.head(10)



"""##### Duplicate Data

Dalam bagian ini, terdapat 1284 data yang duplikat
"""

duplicates = phising_df[phising_df.duplicated()]
duplicates.shape

"""##### Outlier"""

check_outlier_features = [feature for feature in phising_df.columns if feature not in ['qty_dot_url', 'domain_in_ip', 'tld_present_params', 'server_client_domain', 'email_in_url', 'domain_spf', 'tls_ssl_certificate', 'url_shortened', 'phishing']]

print(check_outlier_features)

"""Dari sini dapat kita lihat bahwa kolom numerik tidak memiliki outlier"""

outlier_indices = detect_outliers(phising_df, check_outlier_features)

print(outlier_indices)

"""##### Imbalance

Indicator: max_values_count and second max_values_count gap should not more than 10% of max_values_count
"""

features = phising_df.columns.tolist()

columns_to_remove = ['asn_ip', 'ttl_hostname','length_url', 'directory_length', 'params_length', 'time_response', 'time_domain_activation', 'time_domain_expiration']

filtered_features = [feature for feature in features if feature not in columns_to_remove]
print(len(filtered_features))

for i in range(len(filtered_features)//2):
  print()
  print(filtered_features[i])
  print(phising_df[filtered_features[i]].value_counts())

for i in range(len(filtered_features)//2, len(filtered_features)):
  print()
  print(filtered_features[i])
  print(phising_df[filtered_features[i]].value_counts())

"""Berikut adalah fitur yang imbalance yang bisa tidak dipakai"""

features_imbalance = [
    "qty_underline_url", "qty_questionmark_url", "qty_equal_url", "qty_at_url", "qty_and_url",
    "qty_exclamation_url", "qty_space_url", "qty_tilde_url", "qty_comma_url", "qty_plus_url",
    "qty_asterisk_url", "qty_hashtag_url", "qty_dollar_url", "qty_percent_url", "qty_tld_url",
    "qty_hyphen_domain", "qty_underline_domain", "qty_slash_domain", "qty_questionmark_domain",
    "qty_equal_domain", "qty_at_domain", "qty_and_domain", "qty_exclamation_domain", "qty_space_domain",
    "qty_tilde_domain", "qty_comma_domain", "qty_plus_domain", "qty_asterisk_domain", "qty_hashtag_domain",
    "qty_dollar_domain", "qty_percent_domain", "domain_in_ip", "server_client_domain",
    "qty_questionmark_directory", "qty_equal_directory", "qty_at_directory", "qty_and_directory",
    "qty_exclamation_directory", "qty_space_directory", "qty_tilde_directory", "qty_comma_directory",
    "qty_plus_directory", "qty_asterisk_directory", "qty_hashtag_directory", "qty_dollar_directory",
    "qty_percent_directory", "qty_hyphen_file", "qty_underline_file", "qty_slash_file",
    "qty_questionmark_file", "qty_equal_file", "qty_at_file", "qty_and_file", "qty_exclamation_file",
    "qty_space_file", "qty_tilde_file", "qty_comma_file", "qty_plus_file", "qty_asterisk_file",
    "qty_hashtag_file", "qty_dollar_file", "qty_percent_file", "qty_slash_params",
    "qty_exclamation_params", "qty_space_params", "qty_tilde_params", "qty_comma_params",
    "qty_plus_params", "qty_asterisk_params", "qty_hashtag_params", "qty_dollar_params",
    "qty_percent_params", "email_in_url", "url_google_index", "domain_google_index", "url_shortened"
]

print(len(features_imbalance))

"""#### Data Correlation

NB: Data masih tidak dapat dicari korelasinya karena masih terdapat banyak data yang kotor pada tiap fitur.
"""

phising_df.corr()

"""### Data Validation

#### Features To Drop

The count feature that can be dropped is 93 taken from imbalanced and too noisy data
"""

# Feature that can be dropped
features_to_drop = set(columns_to_drop) | set(features_imbalance) | set(feature_error_type)
print("This is feature that can be dropped:", features_to_drop)
print(len(features_to_drop))

"""#### Records To Drop"""

row_to_drop_indices = list(set(records_with_nan_values.index) | set(records_with_negative_values.index) | set(non_float_records_index))
print(len(row_to_drop_indices))

"""## Data Preparation"""

phising_df_preparation = phising_df.copy()
phising_df_preparation.head(10)

"""### Data Selection

#### Feature Selection
"""

phising_df_preparation = phising_df_preparation.drop(columns=features_to_drop)
phising_df_preparation.head(10)

phising_df_preparation.shape

"""#### Record Selection"""

phising_df_preparation = phising_df_preparation.drop(row_to_drop_indices)
print(phising_df_preparation.shape)
phising_df_preparation.head(10)

"""### Data Improvement

#### Remaining nan records
"""

phising_df_preparation.isna().sum()

"""There is no more nan records in every selected features

#### Duplicate Value
"""

# Checking for duplicate rows
duplicate_rows = phising_df_preparation[phising_df_preparation.duplicated()]
duplicate_rows.shape

# Removing duplicated rows
phising_df_preparation.drop_duplicates(inplace=True)
phising_df_preparation.shape

"""#### Error Records"""

phising_df_preparation[phising_df_preparation < 0].count()

# Checking cardinality for selected features
for column in phising_df_preparation.columns:
  print(column)
  print(phising_df_preparation[column].unique())

median_data_replacement = ['qty_redirects', 'domain_spf', 'qty_ip_resolved']

# median replacement
for feature in median_data_replacement:
    negative_condition = phising_df_preparation[feature] < 0
    phising_df_preparation.loc[negative_condition, feature] = phising_df_preparation[feature].median()

phising_df_preparation[phising_df_preparation < 0].count()

"""### Data Construction

#### Label(y) and train(X) data
"""

X = phising_df_preparation.drop('phishing', axis=1)
X.shape

y = phising_df_preparation['phishing']
y.shape

"""#### Data Representation"""

# Data normalization
sc = StandardScaler()
X = sc.fit_transform(X)

"""## Modeling

### Spliting test and train
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""### Model"""

model = svm.SVC()

model.fit(X_train, y_train)

"""#### Prediction"""

predict_label = model.predict(X_test)

"""## Evaluation"""

accuracy = accuracy_score(y_test, predict_label)
print(f'Test Accuracy: {accuracy*100}%')