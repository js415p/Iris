# %% [markdown]
# # 붓꽃 데이터 분석

# %%
### 필요한 라이브러리 import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## 데이터 불러오기

# %%
### 데이터프레임 생성

# 파일 경로 설정
file_path = '/content/drive/MyDrive/데이터 분석/iris_dataset.csv'

# pd.read_csv() 함수 사용
df_iris = pd.read_csv(file_path)

# 결과 확인하기
print(df_iris)

# %% [markdown]
# ## 데이터 전처리

# %%
### 누락 데이터 처리

# 각 컬럼별 누락 데이터의 수 확인
num_nulls = df_iris.isnull().sum()

# 결과 확인
print(f'각 컬럼별 누락 데이터의 수 : \n{num_nulls}')

# %%
### 이상치 처리

# 각 컬럼별 이상치 확인 --> boxplot 이용
df_iris.plot(kind='box', rot=45)
plt.show()

# %% [markdown]
# ## 데이터 탐색

# %%
### 붓꽃 품종의 종류별 개수 확인 --> '정답' 분포 확인 (불균형 또는 균형)

# 정답의 빈도수 추출
counts = df_iris.loc[:,'label'].value_counts()
print(f'붓꽃 데이터의 품종별 개수 확인 : \n{counts}')

print('-'*80)

# 품종별 분포 시각화 --> 범주형 데이터의 빈도수 시각화 (1)
counts.plot(kind='bar', rot=45)
plt.show()

print('-'*80)

# 품종별 분포 시각화 --> 범주형 데이터의 빈도수 시각화 (2)
sns.countplot(df_iris, x='label')
plt.show()

# %% [markdown]
# ### 각 특성별 품종 간의 차이 시각화

# %%
### 컬럼 이름 추출
names = df_iris.columns
print(f'각 컬럼의 이름 : \n{names}')

# %% [markdown]
# #### 꽃받침의 길이와 품종 간의 차이 시각화

# %%
### 꽃받침의 길이의 분포 시각화
sns.kdeplot(data=df_iris, x='sepal length (cm)', hue='label')
plt.show()

# %% [markdown]
# #### 꽃받침의 너비와 품종 간의 차이 시각화

# %%
### 꽃받침의 너비 분포 시각화 --> 연속형 --> kdeplot

# 그래프 생성 및 출력
sns.kdeplot(data=df_iris, x=names[1], hue=names[-1])
plt.show()

# %% [markdown]
# #### 꽃잎의 길이와 품종 간의 차이 시각화

# %%
### 꽃잎의 길이 분포 시각화 --> 연속형 --> kdeplot

# 그래프 생성 및 출력
sns.kdeplot(data=df_iris, x=names[2], hue=names[-1])
plt.show()

# %% [markdown]
# #### 꽃잎의 너비와 품종 간의 차이 시각화

# %%
### 꽃잎의 너비 분포 시각화 --> 연속형 --> kdeplot

# 그래프 생성 및 출력
sns.kdeplot(data=df_iris, x=names[3], hue=names[-1])
plt.show()

# %% [markdown]
# #### 꽃잎의 길이와 너비의 관계 시각화

# %%
### scatterplot 사용
sns.scatterplot(data=df_iris, x=names[2], y=names[3], hue=names[-1])
plt.show()

# %%
### scatterplot 시각화
sns.scatterplot(data=df_iris, x=names[0], y=names[1], hue=names[-1])
plt.show()


