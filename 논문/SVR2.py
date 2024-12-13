import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import numpy as np

# CSV 파일 불러오기
csv_path = 'normalized_with_season.csv'
data = pd.read_csv(csv_path, encoding='cp949')

# '월' 열에서 계절 정보 추가
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

if '월' in data.columns:
    data['Month'] = data['월'] % 100
    data['Season'] = data['Month'].apply(get_season)
else:
    raise KeyError("'월' 열이 없습니다. CSV 파일을 확인하세요.")

# 결측값 확인 및 처리
if data.isnull().sum().sum() > 0:
    print("데이터에 결측값이 있습니다. 결측값 처리가 필요합니다.")
    data = data.dropna()

# SVR 분석 함수
def analyze_pollutant_with_svr_cv(data, season, pollutant, conditions, n_splits=5):
    # 선택한 계절 데이터 필터링
    season_data = data[data['Season'] == season]

    # 독립변수와 종속변수 설정
    X = season_data[conditions]
    y = season_data[pollutant]

    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SVR 모델 정의
    model = SVR(kernel='linear')

    # KFold 설정
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 교차 검증 수행
    r2_scores, mae_scores, rmse_scores = [], [], []
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 학습 및 예측
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 성능 지표 저장
        r2_scores.append(r2_score(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    # 평균 성능 지표 계산
    mean_r2 = np.mean(r2_scores)
    mean_mae = np.mean(mae_scores)
    mean_rmse = np.mean(rmse_scores)

    # 중요도 계산
    if model.kernel == 'linear':
        coefficients = model.coef_.flatten()
        total_importance = np.sum(np.abs(coefficients))
        importance_percent = [(condition, abs(coef) / total_importance * 100)
                              for condition, coef in zip(conditions, coefficients)]
        sorted_importance = sorted(importance_percent, key=lambda x: x[1], reverse=True)
    else:
        raise AttributeError("중요도 계산은 'linear' 커널에서만 가능합니다.")

    # 결과 출력
    print(f"--- {season} ---")
    print(f"대기오염물질: {pollutant}")
    print(f"기상조건: {', '.join(conditions)}")
    print(f"평균 R2: {mean_r2:.4f}")
    print(f"평균 MAE: {mean_mae:.4f}")
    print(f"평균 RMSE: {mean_rmse:.4f}")
    print()

    print("기상조건 중요도:")
    for condition, importance in sorted_importance:
        print(f"  {condition}: {importance:.2f}%")

    return {
        'Mean R2': mean_r2,
        'Mean MAE': mean_mae,
        'Mean RMSE': mean_rmse,
        'Importance': sorted_importance
    }

# 사용 예시
selected_season = 'Spring'
selected_pollutant = 'CO'
selected_conditions = ['평균기온(℃)', '최저기온(℃)']

result = analyze_pollutant_with_svr_cv(data, selected_season, selected_pollutant, selected_conditions)
