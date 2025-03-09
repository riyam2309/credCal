import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Sidebar for Navigation
st.sidebar.title("Dashboard")
page = st.sidebar.radio("Select Page", ["Business Risk Analysis", "New and Budding Startups"])

# Load the Data
data = pd.read_csv("./50_Startups.csv")

# Calculate Min, Max, and Range of Profit
min_profit = data['Profit'].min()
max_profit = data['Profit'].max()
profit_range = max_profit - min_profit

# Function to Assign Risk Category
def assign_risk(profit):
    if profit <= 74248.21:
        return 'High Risk'
    elif profit <= 133815.02:
        return 'Medium Risk'
    else:
        return 'Low Risk'

# Apply Function to DataFrame
data['Risk_Category'] = data['Profit'].apply(assign_risk)

# Styling
st.markdown("""
<style>
.big-font { font-size:22px !important; color: #2E86C1; font-weight: bold; }
.title-font { font-size:35px !important; text-align: center; color: #1A5276; font-weight: bold; background-color: #D4E6F1; padding: 10px; border-radius: 10px; }
.button-style { background-color: #117A65; color: white; padding: 10px; border-radius: 10px; text-align: center; }
.text-style { font-size:18px !important; color: #283747; }
</style>
""", unsafe_allow_html=True)

if page == "Business Risk Analysis":
   st.switch_page("./new.py") 

elif page == "New and Budding Startups":
    st.markdown("<p class='title-font'>New and Budding Startups</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='big-font'>Minimum Profit: {min_profit}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='big-font'>Maximum Profit: {max_profit}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='big-font'>Range of Profit: {profit_range}</p>", unsafe_allow_html=True)

    user_profit = st.number_input("Enter your Profit value:", min_value=0.0, step=1000.0, key='new_startup_input')
    if st.button("Submit", key='new_startup_button'):
        user_risk = assign_risk(user_profit)
        st.success(f"You belong to '{user_risk}' category")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='R&D Spend', y='Profit', hue='Risk_Category', data=data, palette={'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'}, alpha=0.7, ax=ax)
        ax.scatter(data['R&D Spend'].mean(), user_profit, color='blue', s=200, label='You Stand Here', edgecolors='black', marker='X')
        ax.text(data['R&D Spend'].mean(), user_profit, f'  Your Profit: {user_profit}', fontsize=12, color='blue')
        ax.set_title('Business Risk Clusters with Your Position', fontsize=16, color='#1A5276')
        ax.set_xlabel('R&D Spend', fontsize=14, color='#283747')
        ax.set_ylabel('Profit', fontsize=14, color='#283747')
        ax.legend(title='Risk Category')
        ax.grid(True)
        st.pyplot(fig)