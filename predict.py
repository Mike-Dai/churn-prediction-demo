import joblib
import pandas as pd
import numpy as np

# 1. 加载模型 (就像读取游戏存档)
model = joblib.load('churn_model.pkl')
print("模型加载成功！")

# 2. 准备新数据 (模拟两个玩家)
# 注意：特征的顺序必须和训练时一模一样！
# 训练时的特征顺序是: 
# ['days_since_reg', 'days_since_last_login', 'sessions_last_7d', 'level', 'total_spent', 'is_payer']

new_players = [
    # 玩家 A: 刚刚注册30天，昨天刚登录，最近玩了20次，等级15，充了50块 (活跃玩家)
    [30, 1, 20, 15, 50.0, 1], 
    
    # 玩家 B: 注册300天了，30天没登录，最近0次游戏，等级5，没充钱 (流失预警)
    [300, 30, 0, 5, 0.0, 0]
]

# 转换成 DataFrame (为了保持格式规范，建议加上列名)
columns = ['days_since_reg', 'days_since_last_login', 'sessions_last_7d', 'level', 'total_spent', 'is_payer']
X_new = pd.DataFrame(new_players, columns=columns)

# 3. 进行预测

# 方式一: 直接预测结果 (0 或 1)
predictions = model.predict(X_new)

# 方式二: 预测流失概率 (0% - 100%) -> 这个更有用！
probs = model.predict_proba(X_new)[:, 1]  # 取出类别'1'(流失)的概率

# 4. 打印结果
for i, player in enumerate(new_players):
    is_churn = "流失" if predictions[i] == 1 else "留存"
    risk_score = probs[i] * 100
    
    print(f"玩家 {i+1} 预测结果: {is_churn}")
    print(f"流失概率: {risk_score:.2f}%")
    print("-" * 30)