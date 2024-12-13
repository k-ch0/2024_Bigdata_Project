import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 'Malgun Gothic' 사용
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 결과 저장 폴더 생성
output_folder = "visualization_results"
os.makedirs(output_folder, exist_ok=True)

# 데이터 불러오기
data = pd.read_csv('normalized_air_weather.csv', encoding='cp949')

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

# 기상조건 및 대기오염물질 컬럼 설정
pollutants = ['PM25', 'PM10', 'SO2', 'O3', 'NO2', 'CO']
weather_conditions = [
    '평균기온(℃)', '평균최고기온(℃)', '최고기온(℃)', '평균최저기온(℃)', '최저기온(℃)', 
    '강수량(mm)', '일최다강수량(mm)', '평균풍속(m/s)', '최대풍속(m/s)', 
    '최대풍속풍향(deg)', '최대순간풍속(m/s)', '최대순간풍속풍향(deg)', 
    '일조합(hr)', '일조율(%)', '평균습도(%rh)', '최저습도(%rh)'
]

# 결과 저장할 딕셔너리
results = {}

# 계절별로 처리
for season in ['Spring', 'Summer', 'Fall', 'Winter']:
    season_data = data[data['Season'] == season]
    scaler = StandardScaler()  # 데이터 표준화

    # 독립변수 (기상 조건) 스케일링
    X = scaler.fit_transform(season_data[weather_conditions])
    
    for pollutant in pollutants:
        y = season_data[pollutant]  # 종속변수
        
        # 선형회귀 모델 학습
        model = LinearRegression()
        model.fit(X, y)
        
        # 중요도 계산
        importance = model.coef_
        
        # 중요도의 절대값을 기반으로 퍼센트 계산
        total_importance = sum(abs(importance))
        importance_percent = [(condition, abs(imp) / total_importance * 100) 
                              for condition, imp in zip(weather_conditions, importance)]
        
        # 중요도 내림차순 정렬
        sorted_importance = sorted(importance_percent, key=lambda x: x[1], reverse=True)
        
        # 결과 저장
        if season not in results:
            results[season] = {}
        results[season][pollutant] = sorted_importance

        # 시각화
        conditions, coefficients = zip(*sorted_importance)
        plt.figure(figsize=(10, 6))
        plt.barh(conditions, coefficients, color='skyblue')
        plt.xlabel('중요도 (%)', fontsize=12)
        plt.ylabel('기상 조건', fontsize=12)
        
        # PM25를 PM2.5로 변환하여 제목 설정
        pollutant_display = pollutant.replace('PM25', 'PM2.5')
        plt.title(f'{season} - {pollutant_display} 중요도', fontsize=14)
        
        plt.gca().invert_yaxis()  # 상위 항목이 위로 오도록 역순 정렬
        
        # 그래프 저장
        save_path = os.path.join(output_folder, f'{season}_{pollutant}_importance_percent.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

# 확인 메시지
print(f"시각화 완료! 그래프는 '{output_folder}' 폴더에 저장되었습니다.")
