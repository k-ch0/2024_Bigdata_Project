import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from matplotlib import rc

# 한글 폰트 설정
rc('font', family='Malgun Gothic')  # Windows에서는 'Malgun Gothic', macOS에서는 'AppleGothic'을 사용
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 불러오기
file_path = 'finaldata.csv'  # 파일 경로를 입력하세요.
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# 열 이름 디코딩 (필요한 경우)
data.columns = [col.encode('ISO-8859-1').decode('cp949') for col in data.columns]

# 대상 변수와 피처 선택
target = 'SO2'
features = data.select_dtypes(include=[float, int]).drop(columns=[target]).columns

# 결측치 제거
data_cleaned = data.dropna()

# 입력 데이터와 레이블 분리
X = data_cleaned[features]
y = data_cleaned[target]

# 데이터 분리 (훈련 및 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 학습
xgb_model = XGBRegressor(random_state=42, n_estimators=100, max_depth=4, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# 변수 중요도 계산
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 변수 중요도 시각화
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('중요도')
plt.ylabel('변수')
plt.title('SO2 변수 중요도 (XGBoost)')
plt.gca().invert_yaxis()
plt.show()

# 변수 중요도 출력
print(feature_importance)
