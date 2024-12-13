import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Matplotlib 한글 폰트 설정
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 'Malgun Gothic' 사용
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# CSV 파일 불러오기
csv_path = 'normalized_air_weather.csv'  # CSV 파일 경로
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

# '월'에서 년도 제거 후 계절 추가
data['Month'] = data['월'] % 100  # 월만 추출
data['Season'] = data['Month'].apply(get_season)

# 선형 회귀 분석 함수
def analyze_pollutant(data, season, pollutant, conditions):
    """
    특정 계절, 대기오염물질, 기상조건을 선택하여 선형회귀 분석 수행 및 결과 출력.
    
    Args:
        data (DataFrame): 데이터셋.
        season (str): 분석할 계절 ('Spring', 'Summer', 'Fall', 'Winter').
        pollutant (str): 대기오염물질 ('PM25', 'PM10', 'SO2', 'O3', 'NO2', 'CO').
        conditions (list): 분석할 기상조건 (컬럼명 리스트).

    Returns:
        dict: 성능 지표 및 회귀 계수.
    """
    # 선택한 계절 데이터 필터링
    season_data = data[data['Season'] == season]
    
    # 독립변수와 종속변수 설정
    X = season_data[conditions]
    y = season_data[pollutant]
    
    # 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 예측
    y_pred = model.predict(X)
    
    # 성능 지표 계산
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)  # MAE 추가
    rmse = np.sqrt(mse)
    
    # 결과 출력
    print(f"--- {season} ---")
    print(f"대기오염물질: {pollutant}")
    print(f"기상조건: {', '.join(conditions)}")
    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")  # MAE 출력
    print(f"RMSE: {rmse:.4f}")
    print()
    
    # 회귀 계수 출력
    coefficients = dict(zip(conditions, model.coef_))
    print("회귀 계수:")
    for condition, coef in coefficients.items():
        print(f"  {condition}: {coef:.4f}")
    
    # 결과 반환
    return {
        'R2': r2,
        'MAE': mae,  # MAE 반환
        'RMSE': rmse,
        'Coefficients': coefficients
    }

# 사용 예시
# 데이터셋, 계절, 대기오염물질, 기상조건을 설정하세요.
selected_season = 'Winter'
selected_pollutant = 'SO2'
selected_conditions = ['평균최저기온(℃)', '평균기온(℃)']  # 원하는 기상조건

result = analyze_pollutant(data, selected_season, selected_pollutant, selected_conditions)