import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. 读取新生成的合理数据
data = pd.read_csv('new_players_data.csv')

# 2. 简单的特征选择
# 我们直接用数字特征，不需要再转换日期了，因为生成时已经做好了 days_since_reg
features = ['days_since_reg', 'days_since_last_login', 'sessions_last_7d', 'level', 'total_spent', 'is_payer']
X = data[features]
y = data['churn_next_7d']

# 3. 训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 不需要 class_weight='balanced' 了，因为生成逻辑里已经控制了比较合理的流失比
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. 结果肯定会变好，因为数据里有“规律”了
print(classification_report(y_test, model.predict(X_test)))

# 看看哪个特征最重要
importance = pd.DataFrame({'feature': features, 'coef': model.coef_[0]})
print(importance.sort_values(by='coef'))

import joblib

# 假设 model 是你刚刚训练好的 LogisticRegression 对象
# 将模型保存为 'churn_model.pkl'
joblib.dump(model, 'churn_model.pkl')

print("模型已保存为 churn_model.pkl，可以去文件夹里查看了！")