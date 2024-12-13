import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
from matplotlib import rc

# 한글 폰트 설정
rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False  

# 데이터 불러오기
file_path = 'C:/finaldata.csv'
data = pd.read_csv(file_path, encoding='cp949')

# 제외할 컬럼 리스트
columns_to_exclude = ['시도', '시군구', '월', '대중교통', '자동차등록수', '공원면적', 'PM10', 'SO2', 'CO', 'PM25', 'O3']
data = data.drop(columns=[col for col in columns_to_exclude if col in data.columns])

# 계절별로 데이터 분리
seasons = data['계절'].unique()
feature_importances = {}

# 계절별 RandomForest 모델 학습 및 특성 중요도 추출
for season in seasons:
    # 계절별 데이터 필터링
    season_data = data[data['계절'] == season]
    
    # 대상 변수와 피처 선택
    target = 'NO2'
    features = season_data.drop(columns=['계절', target]).columns
    X = season_data[features]
    y = season_data[target]
    
    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # RandomForest 모델 학습
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=4)
    rf_model.fit(X_train, y_train)
    
    # 특성 중요도 저장
    feature_importances[season] = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

# 시각화: 계절별 특성 중요도
plt.figure(figsize=(15, 10))
for i, season in enumerate(seasons, 1):
    plt.subplot(2, 2, i)  # 4개 계절에 맞게 서브플롯 생성
    sns.barplot(data=feature_importances[season], x='Importance', y='Feature')
    plt.title(f'NO2 Feature Importance ({season})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

plt.show()

# 5-fold 교차검증 함수 (계절별 교차검증)
def cross_validate_model(season, selected_features, target):
    season_data = data[data['계절'] == season]
    X = season_data[selected_features]
    y = season_data[target]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=4)

    rmse_scores = []
    r2_scores = []
    mae_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse_scores.append(sqrt(mean_squared_error(y_test, y_pred)))
        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))

    print(f"\n5-Fold Cross-Validation Results for {season} season")
    print(f"Average RMSE: {sum(rmse_scores) / len(rmse_scores):.4f}")
    print(f"Average R²: {sum(r2_scores) / len(r2_scores):.4f}")
    print(f"Average MAE: {sum(mae_scores) / len(mae_scores):.4f}")

# 각 계절에 대한 교차검증 수행
cross_validate_model('겨울', ['최대순간풍속(m/s)', '평균습도(%rh)'], 'NO2')
cross_validate_model('여름', ['평균습도(%rh)', '일조합(hr)'], 'NO2')
cross_validate_model('가을', ['최고기온(℃)', '평균최고기온(℃)'], 'NO2')
cross_validate_model('봄', ['평균최고기온(℃)', '최대순간풍속(m/s)'], 'NO2')