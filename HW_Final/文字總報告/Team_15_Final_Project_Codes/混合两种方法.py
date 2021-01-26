import pandas as pd
import math
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

time_start=time.time()
data = pd.read_csv("test_data4.csv")

data = data.sort_values(by=['ID'])
data_x = data.drop(["乳量","ID"],axis=1)
data_y = data["乳量"]
data_test = data_x[data["乳量"] < 0]
data_train_x = data_x[data["乳量"] >= 0]
data_train_y = data_y[data["乳量"] >= 0]
X_train, X_test, y_train, y_test = train_test_split(data_train_x,data_train_y,random_state=10, test_size=0.3)

n_estimators = 300
from sklearn.ensemble import ExtraTreesRegressor
model1 = ExtraTreesRegressor(random_state=10, n_estimators=n_estimators)
model1.fit(X_train,y_train)

importances = model1.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%d.  %s   (%.5f)" % (f + 1,X_train.columns[indices[f]], importances[indices[f]]))

pre1 = model1.predict(X_test)
print("R^2: %.3f, RMSE: %.3f"%(r2_score(pre1, y_test),math.sqrt(mean_squared_error(pre1,y_test))))


from sklearn.ensemble import GradientBoostingRegressor
model3 =  GradientBoostingRegressor(random_state=10, max_depth=6, n_estimators=n_estimators)
model3.fit(X_train,y_train)
pre3 = model3.predict(X_test)
print("R^2: %.3f, RMSE: %.3f"%(r2_score(pre3, y_test),math.sqrt(mean_squared_error(pre3,y_test))))

pre = (pre1 +  pre3) / 2
print("混合后R^2: %.3f, RMSE: %.3f"%(r2_score(pre, y_test),math.sqrt(mean_squared_error(pre,y_test))))



pre1 = model1.predict(data_test)
pre3 = model3.predict(data_test)

pre = (pre1 +  pre3) / 2
pd.DataFrame({"ID": pd.read_csv("data/submission.csv")["ID"], "1":pre}).to_csv("0.3兩個模型混合.csv", index=False)
time_end=time.time()
print('totally cost',time_end-time_start)

