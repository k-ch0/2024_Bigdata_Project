import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the uploaded CSV file
file_path = 'normalized_with_season.csv'
data = pd.read_csv(file_path, encoding='euc-kr')

# Verify the structure of the data
data.head(), data.columns
from sklearn.metrics import accuracy_score

# 제외하고 싶은 열들의 리스트
exclude_cols = []

# 시작 열과 종료 열 지정 (제거할 열 제외하고 포함할 열 범위 설정)
start_col = '대중교통'
end_col = '공원면적'

# 열 범위 선택 후 제외할 열 제거
columns_for_clustering = data.loc[:, start_col:end_col].drop(columns=exclude_cols)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(columns_for_clustering)

# Perform KMeans clustering with k=4
kmeans = KMeans(n_clusters=4, random_state=42)
data['클러스터'] = kmeans.fit_predict(scaled_data)

# Evaluate accuracy per season
accuracy_by_season = {}
for season in data['계절'].unique():
    season_data = data[data['계절'] == season]
    true_labels = season_data['계절']
    predicted_clusters = season_data['클러스터']
    # Calculate season-wise accuracy as the percentage of the most common cluster
    accuracy_by_season[season] = (predicted_clusters.value_counts().max() / len(predicted_clusters))

# Convert the results to a DataFrame for better visualization
accuracy_df = pd.DataFrame(list(accuracy_by_season.items()), columns=['계절', '정확도'])
print(accuracy_df)