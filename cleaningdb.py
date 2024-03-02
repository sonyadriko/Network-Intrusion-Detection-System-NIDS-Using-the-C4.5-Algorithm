import pandas as pd

# Load your dataset
# df = pd.read_excel('dataset_lengkap.xlsx')
sheet_names = pd.ExcelFile('dataset_lengkap.xlsx').sheet_names
print(sheet_names)
df = pd.read_excel('dataset_lengkap.xlsx')


# Handling missing values
df = df.dropna()

# Encoding categorical variables
df = pd.get_dummies(df, columns=['proto', 'service'])

# Scaling numeric features (using Min-Max scaling in this example)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']] = scaler.fit_transform(df[['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']])

# Removing irrelevant features (if needed)
# df = df.drop(['irrelevant_feature1', 'irrelevant_feature2'], axis=1)

# Handling outliers (if needed)

# Save the cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)