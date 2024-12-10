import shap
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# 데이터 불러오기
file_path = "normalized_with_season.csv"  # 파일 경로를 지정하세요.
data = pd.read_csv(file_path, encoding='latin1', header=0, names=[
    "City", "District", "Month", "PublicTransport", "PM25", "PM10", "SO2", "O3", "NO2", "CO", 
    "AvgTemp(℃)", "AvgMaxTemp(℃)", "MaxTemp(℃)", "AvgMinTemp(℃)", "MinTemp(℃)", 
    "Precipitation(mm)", "MaxDailyPrecipitation(mm)", "AvgWindSpeed(m/s)", 
    "MaxWindSpeed(m/s)", "MaxWindDirection(deg)", 
    "MaxGustSpeed(m/s)", "MaxGustDirection(deg)", "SunHours(hr)", 
    "SunlightRate(%)", "AvgHumidity(%rh)", "MinHumidity(%rh)", 
    "CarRegistration", "ParkArea", "Season"
])

# 특성과 타겟 분리
X = data.drop(columns=["PM10", "City", "District", "Season"])
y = data["PM10"]

# 결측값 처리
X = SimpleImputer(strategy="mean").fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# SHAP 분석
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 변수 중요도 시각화
shap.summary_plot(shap_values, X_test, feature_names=data.drop(columns=["PM10", "City", "District", "Season"]).columns)
