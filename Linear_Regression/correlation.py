import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = "C:/Windows/Fonts/malgun.ttf"  # 폰트 경로를 지정해줘야 할 수 있음
font_prop = font_manager.FontProperties(fname=font_path)

df = pd.read_csv('normalized_with_season.csv', encoding='cp949')
df = df.replace({',': ''}, regex=True)
numeric_columns = df.select_dtypes(include=['object']).columns  # 문자열로 되어있는 컬럼만 선택
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

df = pd.get_dummies(df, columns=['시도', '시군구'], drop_first=True)

X = df.drop(columns=['PM25', 'PM10', 'SO2', 'O3', 'NO2', 'CO', '월'])  # '월' 컬럼을 제외
y = df[['PM25', 'PM10', 'SO2', 'O3', 'NO2', 'CO']]  # 예측할 대기 오염 물질들

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 대기 오염 물질들에 대한 모델 학습 및 중요도 계산
feature_importances = {}

for pollutant in ['PM25', 'PM10', 'SO2', 'O3', 'NO2', 'CO']:
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train[pollutant])  # 각 물질에 대해 모델 학습
    
    feature_importances[pollutant] = rf_model.feature_importances_

feature_names = X.columns

for pollutant, importance in feature_importances.items():
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # 변수 중요도 출력
    print(f"{pollutant} Importance:")
    print(importance_df)
    
    # 변수 중요도 시각화
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance', fontproperties=font_prop) 
    plt.title(f'Feature Importance for {pollutant}', fontproperties=font_prop)  
    plt.yticks(rotation=0, fontproperties=font_prop)  
    
    # 이미지 저장
    plt.savefig(f'feature_importance_{pollutant}.png')  # 파일로 저장
    plt.close()  # 현재 플롯을 닫아서 메모리 절약
