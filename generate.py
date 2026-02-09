import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_realistic_data(num_rows=1000):
    data = []
    
    # 设置参考时间点 (假设今天是数据采集日)
    current_date = datetime(2025, 12, 31)
    
    for i in range(num_rows):
        user_id = f"user_{i}"
        
        # 1. 基础属性 (随机)
        gender = random.choice(['Male', 'Female'])
        age = random.randint(16, 50)
        device = random.choice(['iOS', 'Android'])
        province = random.choice(['Guangdong', 'Beijing', 'Shanghai', 'Sichuan', 'Zhejiang'])
        
        # 2. 生成具有逻辑的时间数据
        # 注册时间：过去 1 年内
        days_since_reg = random.randint(1, 365)
        registration_date = current_date - timedelta(days=days_since_reg)
        
        # 3. 生成用户类型 (User Persona) - 这是数据合理的关键
        # 我们随机设定用户的“沉迷程度”：0-0.3(容易流失), 0.3-0.7(普通), 0.7-1.0(铁粉)
        engagement_factor = random.random() 
        
        # 4. 根据沉迷程度生成行为特征
        
        # A. 上次登录时间 (Recency): 铁粉最近肯定登录过，流失用户很久没登了
        if engagement_factor > 0.8:
            days_since_last_login = random.randint(0, 2) # 铁粉：最近3天内登过
        elif engagement_factor > 0.4:
            days_since_last_login = random.randint(3, 10) # 普通：最近10天内
        else:
            days_since_last_login = random.randint(11, 60) # 潜在流失：很久没登了
            
        last_login_date = current_date - timedelta(days=days_since_last_login)
        
        # B. 游戏频次 (Frequency): 过去7天登录次数
        if days_since_last_login > 7:
            sessions_last_7d = 0 # 超过7天没登，最近当然是0次
        else:
            # 越沉迷，登录越频繁
            base_sessions = int(engagement_factor * 20) # 最多20次
            sessions_last_7d = max(1, base_sessions + random.randint(-2, 2))
            
        # C. 等级与时长 (Level & Duration)
        # 注册越久 + 越沉迷 -> 等级越高
        avg_playtime_per_day = engagement_factor * 120 # 最多每天玩2小时
        total_playtime_hours = (days_since_reg * avg_playtime_per_day / 60) * random.uniform(0.8, 1.2)
        level = int(total_playtime_hours / 5) + 1 # 假设每5小时升一级
        level = min(level, 100) # 封顶100级
        
        # D. 付费情况 (Monetary)
        # 只有沉迷度高的人才倾向于付费
        total_spent = 0
        is_payer = 0
        if engagement_factor > 0.6 and random.random() > 0.5:
            is_payer = 1
            # 简单的二八定律：大部分人花小钱，少部分人花大钱
            if random.random() > 0.9: 
                total_spent = random.uniform(500, 5000) # 大R
            else:
                total_spent = random.uniform(6, 200) # 小R
                
        # 5. 生成最关键的 Label: churn_next_7d (未来7天是否流失)
        # 我们基于上面的特征计算一个“流失概率”，然后掷骰子
        
        churn_probability = 0.0
        
        # 逻辑1: 已经很久没登录的人，几乎肯定流失 (惯性)
        if days_since_last_login > 14:
            churn_probability += 0.95
        # 逻辑2: 最近玩得很频繁的人，不太可能流失
        elif sessions_last_7d > 5:
            churn_probability -= 0.4
        
        # 逻辑3: 充了钱的人，比较难流失 (沉没成本)
        if total_spent > 100:
            churn_probability -= 0.3
            
        # 逻辑4: 等级太低的人容易流失 (新手体验差)
        if level < 5 and days_since_reg > 3:
            churn_probability += 0.4
            
        # 逻辑5: 基础流失率
        churn_probability += 0.2
        
        # 截断概率在 0.05 到 0.95 之间 (保留一点随机性)
        churn_probability = max(0.05, min(0.95, churn_probability))
        
        # 掷骰子决定最终标签
        churn_label = 1 if random.random() < churn_probability else 0
        
        # 6. 整理行数据
        row = {
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'province': province,
            'device_type': device,
            'registration_date': registration_date.strftime('%Y-%m-%d'),
            'last_login_date': last_login_date.strftime('%Y-%m-%d'),
            'days_since_reg': days_since_reg,
            'days_since_last_login': days_since_last_login,
            'sessions_last_7d': sessions_last_7d,
            'level': level,
            'total_spent': round(total_spent, 2),
            'is_payer': is_payer,
            'churn_next_7d': churn_label
        }
        data.append(row)
        
    return pd.DataFrame(data)

# 生成并保存
df = generate_realistic_data(50000)
df.to_csv('new_players_data.csv', index=False)

print("数据生成完毕！")
print(f"流失用户比例: {df['churn_next_7d'].mean():.2%}")
print("\n前5行预览：")
print(df[['user_id', 'days_since_last_login', 'total_spent', 'churn_next_7d']].head())