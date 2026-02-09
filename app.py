import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 页面基础设置 ---
st.set_page_config(page_title="玩家流失预测系统", layout="wide")

# 加载模型 (使用缓存，防止每次刷新都重新加载)
@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')

# 加载数据
@st.cache_data
def load_data():
    return pd.read_csv('new_players_data.csv')

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(f"请确保目录下一句存在 'churn_model.pkl' 和 'new_players_data.csv'。错误信息: {e}")
    st.stop()

# --- 2. 侧边栏：模拟单个玩家 ---
st.sidebar.header("🕵️‍♂️ 玩家行为模拟器")
st.sidebar.markdown("调整下方参数，预测流失概率：")

# 输入特征 (必须和训练时的顺序一致)
# ['days_since_reg', 'days_since_last_login', 'sessions_last_7d', 'level', 'total_spent', 'is_payer']

days_since_reg = st.sidebar.slider("已注册天数", 1, 730, 30)
days_since_last_login = st.sidebar.slider("距离上次登录 (天)", 0, 60, 3)
sessions_last_7d = st.sidebar.number_input("过去7天游戏场次", 0, 100, 5)
level = st.sidebar.slider("玩家等级", 1, 100, 10)
total_spent = st.sidebar.number_input("历史充值金额 ($)", 0.0, 10000.0, 0.0)
is_payer = 1 if total_spent > 0 else 0

# 构造输入数据 DataFrame
input_data = pd.DataFrame({
    'days_since_reg': [days_since_reg],
    'days_since_last_login': [days_since_last_login],
    'sessions_last_7d': [sessions_last_7d],
    'level': [level],
    'total_spent': [total_spent],
    'is_payer': [is_payer]
})

# --- 3. 主界面：预测结果 ---
st.title("🎮 游戏玩家流失预警系统")
st.markdown("### AI 实时预测面板")

col1, col2 = st.columns([1, 2])

with col1:
    # 调用模型预测
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # 获取流失概率

    # 根据概率显示不同颜色和状态
    if probability > 0.7:
        status_color = "red"
        status_text = "高风险流失"
        st.error(f"⚠️ 预测结果：{status_text}")
    elif probability > 0.3:
        status_color = "orange"
        status_text = "潜在风险"
        st.warning(f"⚖️ 预测结果：{status_text}")
    else:
        status_color = "green"
        status_text = "忠诚玩家"
        st.success(f"✅ 预测结果：{status_text}")

    st.metric(label="流失概率", value=f"{probability*100:.2f}%")

with col2:
    st.markdown("#### 💡 运营建议")
    if probability > 0.7:
        st.write("👉 **建议操作**：该用户极大概率在7天内流失。建议立即发送 **召回短信** 或 **赠送限时回归礼包**。")
    elif probability > 0.3:
        st.write("👉 **建议操作**：用户活跃度下降。建议通过 **推送通知** 提醒其参加当前的周末活动。")
    else:
        st.write("👉 **建议操作**：用户非常活跃。建议推荐 **付费活动** 或 **高阶公会** 以提升 LTV（生命周期价值）。")

st.divider()

# --- 4. 数据仪表盘 ---
st.markdown("### 📊 全服数据透视")
tab1, tab2 = st.tabs(["流失关键因素", "数据概览"])

with tab1:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**距离上次登录 vs 流失情况**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='churn_next_7d', y='days_since_last_login', data=df, palette="Set2", ax=ax1)
        ax1.set_xticklabels(['留存', '流失'])
        st.pyplot(fig1)
        
    with col_b:
        st.markdown("**付费金额 vs 流失情况 (仅付费用户)**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        payer_df = df[df['total_spent'] > 0]
        sns.boxplot(x='churn_next_7d', y='total_spent', data=payer_df, palette="Set3", ax=ax2)
        ax2.set_xticklabels(['留存', '流失'])
        st.pyplot(fig2)

with tab2:
    st.dataframe(df.head(10))
    st.caption(f"当前数据集共包含 {len(df)} 名玩家数据。")