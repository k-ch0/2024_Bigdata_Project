import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import numpy as np

# 데이터 불러오기
file_path = 'finaldata.csv'  # 파일 경로를 입력하세요.
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# 열 이름 디코딩 (필요한 경우)
data.columns = [col.encode('ISO-8859-1').decode('cp949') for col in data.columns]

# 대상 변수와 피처 선택
target = 'SO2'  # SO2변수
all_features = data.select_dtypes(include=[float, int]).columns

# "연월" 변수 제거
excluded_features = ['월']  # 제거할 변수 이름
features_with_month = [col for col in all_features if col != target]  # 연월 포함
features_without_month = [col for col in features_with_month if col not in excluded_features]  # 연월 제외

# 결측치 제거
data_cleaned = data.dropna()

# 데이터 분리
X_with_month = data_cleaned[features_with_month]
X_without_month = data_cleaned[features_without_month]
y = data_cleaned[target]

X_train_with, X_test_with, y_train, y_test = train_test_split(X_with_month, y, test_size=0.2, random_state=42)
X_train_without, X_test_without = train_test_split(X_without_month, test_size=0.2, random_state=42)

# XGBoost 모델 학습 및 평가
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = XGBRegressor(random_state=42, n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, mae

# 성능 비교
rmse_with, mae_with = train_and_evaluate(X_train_with, X_test_with, y_train, y_test)
rmse_without, mae_without = train_and_evaluate(X_train_without, X_test_without, y_train, y_test)

# 결과 출력
print(f"With '월' - RMSE: {rmse_with:.4f}, MAE: {mae_with:.4f}")
print(f"Without '월' - RMSE: {rmse_without:.4f}, MAE: {mae_without:.4f}")
