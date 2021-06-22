from scipy.sparse.construct import random
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 보스턴 데이터셋 불러오기
raw_boston = datasets.load_boston()

# 독립변수와 종속변수로 분리
X = raw_boston.data
y = raw_boston.target

# 트레인, 테스트셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

# 표준화 스케일링
std_scaler = StandardScaler()
X_train_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.fit_transform(X_test)

# 학습
clf_linear = LinearRegression()
clf_linear.fit(X_train_scaled, y_train)
clf_linear.fit(X_train, y_train)

# 예측
pred_linear = clf_linear.predict(X_test_scaled)
pred_linear = clf_linear.predict(X_test)

# 평가
mean_squared_error(y_test, pred_linear)

'''
스케일링 전 : 30.77634664423657
스케일링 후 : 29.515137790197734
'''

#######
'''파이프라인 적용'''
# 이전 생략 / 다시 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

# 파이프라인
linear_pipline = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_regression', LinearRegression())
    ])

# 학습
linear_pipline.fit(X_train, y_train)

# 예측
pred_linear = linear_pipline.predict(X_test)

# 평가
mean_squared_error(y_test, pred_linear)