# %% [markdown]
# ## DecisionTree 분류 모델을 이용한 분류

# %% [markdown]
# ### 데이터 불러오기

# %%
### 필요한 라이브러리 import
import pandas as pd
import numpy as np

# %%
### 데이터 불러오기

# 파일 경로 설정
file_path = '/content/drive/MyDrive/데이터 분석/iris_dataset.csv'

# pd.read_csv() 함수 --> DataFrame 자료형 생성
df_iris = pd.read_csv(file_path)

# 결과 확인하기
print(df_iris)

# %% [markdown]
# ### 데이터 전처리

# %%
### 붓꽃 품종의 정답 label --> 숫자 --> label encoding 실행

# 붓꽃 품종 종류 --> 알파벳 순으로 정리 --> np.unique()
arr = df_iris.loc[:,'label'].values
kinds = np.unique(ar=arr)
print(f'붓꽃 품종의 종류 확인 : \n{kinds}')

# replace() 사용 --> label 컬럼의 값 수정 : 문자열 --> 숫자
df_iris.loc[:,'label'] = df_iris.loc[:,'label'].replace({kinds[0]:0,
                                                     kinds[1]:1,
                                                     kinds[2]:2})

# 결과 확인하기
print(df_iris)

# %% [markdown]
# ### 학습용 데이터와 평가용 데이터 생성

# %%
### 80:20의 비율로 학습용 데이터와 평가용 데이터 생성

# 필요한 함수 import
from sklearn.model_selection import train_test_split

# X_data 생성
X_data = df_iris.drop(columns=['label'])
print(f'X 데이터 확인 : \n{X_data}')

print('-'*80)

# Y_data 생성
y_data = df_iris.loc[:,'label']
print(f'y 데이터 확인 : \n{y_data}')

print('-'*80)

# train_test_split() 함수 사용 --> 결과 값 : 4가지
X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    y_data,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y_data)

# %%
### 학습용 데이터 확인

# 데이터의 자료형 확인
print(f'X_train의 자료형 : {type(X_train)}')
print('-'*80)
print(f'y_train의 자료형 : {type(y_train)}')

print('-'*80)

# 데이터의 모양 확인
print(f'X_train의 모양 : {X_train.shape}')
print('-'*80)
print(f'y_train의 모양 : {y_train.shape}')

print('-'*80)

# 데이터의 인덱스 확인
x_index = X_train.index
y_index = y_train.index
print(f'X_train의 인덱스 : \n{x_index}')
print('-'*80)
print(f'y_train의 인덱스 : \n{y_index}')

# %%
### 학습용 데이터 --> 정답의 빈도수 확인
print(y_train.value_counts())

# %% [markdown]
# ### 모델 생성

# %%
### 필요한 함수 import
from sklearn.tree import DecisionTreeClassifier

# %%
### 모델 생성 함수 호출, 모델 생성
dt = DecisionTreeClassifier(random_state=0)

# %% [markdown]
# ### 모델 학습

# %%
dt.fit(X_train, y_train)

# %% [markdown]
# ### 모델 학습 시 생성된 의사 결정 트리 시각화

# %%
### 학습 시 생성된 의사 결정 트리의 깊이 확인
print(dt.get_depth())

# %%
### 학습 시 생성된 의사 결정 트리 시각화

# 필요한 라이브러리 / 함수 import
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 이미지의 크기 재설정
plt.figure(figsize=(12, 8))

# 시각화
plot_tree(dt, feature_names=X_train.columns, filled=True)
plt.show()

# %% [markdown]
# ### 평가용 데이터를 이용한 예측

# %%
### 평가용 데이터를 이용한 예측
pred_test = dt.predict(X_test)
print(f'평가용 데이터에 대한 예측 : \n{pred_test}')

# %%
### 평가용 정답 확인
print(f'평가용 데이터에 대한 정답 : \n{y_test.values}')

# %%
### 평가용 데이터에 대한 예측의 정확도 측정 --> 비교 연산
accuracy = (y_test == pred_test)
num_true = accuracy.sum()
print(f'평가용 데이터에 대해서 맞힌 개수 = {num_true}개')
print('-'*80)
accuracy_score = num_true/y_test.size
print(f'평가용 데이터에 대한 정확도 = {accuracy_score}')

# %% [markdown]
# ### 모델 평가

# %%
# 필요한 함수 import
from sklearn.metrics import accuracy_score

# 평가용 데이터에 대한 정확도 측정(평가)
accuracy = accuracy_score(y_test, pred_test)

# 결과 확인하기
print(f'평가용 데이터에 대한 정확도 = {accuracy}')

# %% [markdown]
# ### GridSearchCV를 이용한 모델 성능 최적화

# %%
### GridSearchCV 함수 실행

# 필요한 함수 import
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# 최적화 할 기본 모델 생성
base_dt = DecisionTreeClassifier(random_state=0)

# 튜닝할 하이퍼파라미터 설정
params = {'max_depth':[3,4,5,6,7,8,9]}

# GridSearchCV 함수 실행, 모델 생성
grid_dt = GridSearchCV(estimator=base_dt,
                       param_grid=params,
                       scoring='accuracy',
                       cv=10)

# 학습 및 평가
grid_dt.fit(X_train, y_train)

# %% [markdown]
# #### 모델 생성

# %%
### 최적의 하이퍼파라미터 확인
print(grid_dt.best_params_)

# %%
### best 모델 생성
best_dt = DecisionTreeClassifier(max_depth=5,
                                 random_state=0)

# %% [markdown]
# #### 모델 학습

# %%
best_dt.fit(X_train, y_train)

# %% [markdown]
# #### 평가용 데이터를 이용한 예측

# %%
pred_test = best_dt.predict(X_test)

# %% [markdown]
# #### 모델 평가

# %%
# 필요한 함수 import
from sklearn.metrics import accuracy_score

# 평가용 데이터를 이용한 정확도 평가
accuracy = accuracy_score(y_test, pred_test)

# 결과 확인하기
print(f'평가용 데이터에 대한 평가 결과 : \n{accuracy}')


