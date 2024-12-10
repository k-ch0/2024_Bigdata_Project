import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats

# Matplotlib 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows에서 'Malgun Gothic' 사용
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 로드 (인코딩을 cp949로 설정)
df = pd.read_csv('normalized_with_season.csv', encoding='cp949')

# '월' 컬럼을 datetime 형식으로 변환
df['date'] = pd.to_datetime(df['월'].astype(str), format='%Y%m')

# 계절 정보 추가 (1~3월: 겨울, 4~6월: 봄, 7~9월: 여름, 10~12월: 가을)
def get_season(month):
    if month in [12, 1, 2]:
        return '겨울'
    elif month in [3, 4, 5]:
        return '봄'
    elif month in [6, 7, 8]:
        return '여름'
    else:
        return '가을'

df['season'] = df['date'].dt.month.apply(get_season)

# '평균기온(℃)', '일조율(%)', 'O3' 컬럼만 추출
df_season = df[['season', '평균기온(℃)', '일조율(%)', 'O3']]

# 계절별로 데이터 나누기
winter = df_season[df_season['season'] == '겨울']
spring = df_season[df_season['season'] == '봄']
summer = df_season[df_season['season'] == '여름']
fall = df_season[df_season['season'] == '가을']

# 이미지 저장 폴더 생성
output_dir = 'C:\\Users\\Owner\\Desktop\\2024_kch\\BigData\\project\\regression\\O3'
os.makedirs(output_dir, exist_ok=True)

# 계절별로 시각화하는 함수 정의
def plot_seasonal_relation(season_data, season_name):
    plt.figure(figsize=(14, 6))
    
    # 평균기온과 O3의 관계 시각화
    plt.subplot(1, 2, 1)
    sns.regplot(x='평균기온(℃)', y='O3', data=season_data, scatter_kws={'s': 50}, line_kws={'color': 'red'})
    corr, p_value = stats.pearsonr(season_data['평균기온(℃)'], season_data['O3'])  # 상관계수, p-value 계산
    slope, intercept, r_value, p_value, std_err = stats.linregress(season_data['평균기온(℃)'], season_data['O3'])  # 기울기 구하기
    plt.title(f'{season_name} - O3 vs 평균기온\n상관계수: {corr:.2f}, 기울기: {slope:.2f}, p-value: {p_value:.4f}', loc='right')

    # y축 범위 확장 (최솟값과 최댓값을 2배로 확장)
    min_y = season_data['O3'].min()
    max_y = season_data['O3'].max()
    plt.ylim(min_y - (max_y - min_y) * 1.0, max_y + (max_y - min_y) * 1.0)  # y축 범위 2배 확장

    # 일조율과 O3의 관계 시각화
    plt.subplot(1, 2, 2)
    sns.regplot(x='일조율(%)', y='O3', data=season_data, scatter_kws={'s': 50}, line_kws={'color': 'blue'})
    corr, p_value = stats.pearsonr(season_data['일조율(%)'], season_data['O3'])  # 상관계수, p-value 계산
    slope, intercept, r_value, p_value, std_err = stats.linregress(season_data['일조율(%)'], season_data['O3'])  # 기울기 구하기
    plt.title(f'{season_name} - O3 vs 일조율\n상관계수: {corr:.2f}, 기울기: {slope:.2f}, p-value: {p_value:.4f}', loc='right')

    # y축 범위 확장 (최솟값과 최댓값을 2배로 확장)
    min_y = season_data['O3'].min()
    max_y = season_data['O3'].max()
    plt.ylim(min_y - (max_y - min_y) * 1.0, max_y + (max_y - min_y) * 1.0)  # y축 범위 2배 확장

    plt.tight_layout()

    # 이미지 저장
    plt.savefig(f'{output_dir}/{season_name}_O3_relation.png')
    plt.close()

# 계절별 상관 행렬 시각화하는 함수 정의
def plot_seasonal_corr(season_data, season_name):
    plt.figure(figsize=(8, 6))
    
    # 상관 행렬 계산
    corr = season_data[['평균기온(℃)', '일조율(%)', 'O3']].corr()

    # 상관 행렬 시각화
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, annot_kws={'size': 12}, cbar_kws={'shrink': 0.8})
    
    # 제목 추가
    plt.title(f'{season_name} - 상관 행렬', fontsize=16)
    
    # 이미지 저장
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{season_name}_correlation_matrix.png')
    plt.close()

# 계절별 회귀 분석 및 상관 행렬 시각화
for season_name, season_data in zip(['겨울', '봄', '여름', '가을'], [winter, spring, summer, fall]):
    plot_seasonal_relation(season_data, season_name)  # 회귀선 시각화
    plot_seasonal_corr(season_data, season_name)  # 상관 행렬 시각화

print("모든 회귀선과 상관 행렬 이미지가 저장되었습니다.")
