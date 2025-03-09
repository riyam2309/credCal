import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import altair as alt
import time
from datetime import datetime
import seaborn as sns

# Load trained model
from cibil_model import rf_model

# Streamlit page setup
st.set_page_config(page_title="CIBIL Score Predictor", layout="wide")


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
if page == "New and Budding Startups":
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
elif page == "Business Risk Analysis":
    st.markdown(
        """
        <style>
            /* Move all Streamlit content to the left and adjust top margin */
            .main {
                text-align: left;
                max-width: 800px;  /* Adjust width */
                margin:  0 ;  /* Add top margin */
                padding-top: 20px;  /* Fine-tune padding */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Custom styles
st.markdown("""
    <style>
        .stSlider, .stNumberInput { width: 80% !important; margin: auto; }
        .card { padding: 20px; border-radius: 10px; background-color: #f9f9f9; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
        .center-text { text-align: center; }
        .error-msg { color: red; font-weight: bold; }
        .stApp {
            background-color: #82CAFF;
            background-size: cover;
            background-position: center;
        }
        /* Button styling */
        .stButton > button {
            display: flex;
            margin: auto;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
        }
        /* Small button */
        .small-button button {
            width: 150px !important;  /* Adjust button width */
            height: 40px !important;  /* Adjust button height */
            font-size: 14px !important;  /* Reduce font size */
            padding: 5px !important;
        }
        /* Fix UI elements */
        input, select, textarea {
            background-color: white !important;
            color: black !important;
        }
        .stNumberInput, .stTextInput, .stSelectbox, .stSlider {
            background: none !important;
            box-shadow: none !important;
        }
        /* Dashboard specific styling */
        .dashboard-card {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #0066cc;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
        }
        .tab-content {
            padding: 20px;
            background-color: white;
            border-radius: 0px 10px 10px 10px;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# Initialize session state for storing historical predictions
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'timestamp', 'annual_revenue', 'loan_amount', 'gst_billing',
        'credit_utilization', 'credit_age', 'repayment_history', 'predicted_score'
    ])

# st.markdown(
#     """
#     <h1 style="text-align: center; color: #00008b; font-size: 52px; font-weight: bold;">
#         CIBIL Score Predictor
#     </h1>
#     <div> <br> </div>
#     <hr>
#     """,
#     unsafe_allow_html=True,
# )

st.markdown(
    """
    <div style="background-color: white; padding: 20px; border-radius: 10px;">
        <h1 style="text-align: center; color: #00008b; font-size: 52px; font-weight: bold;">
            CIBIL Score Predictor
        </h1>
    </div>
    <br>
    <hr>
    """,
    unsafe_allow_html=True,
)


# Create main layout - 25% for input, 75% for gauge
input_col, gauge_col = st.columns([50, 50])  # This gives us the 25/75 split you want

# Input section in the left column (25%)
with input_col:
    sub_col1, sub_col2 = st.columns(2)
    
    with sub_col1:
        with st.container():
            annual_revenue = st.number_input("Annual Revenue (₹)", min_value=500000, step=10000, value=1000000)
            loan_amount = st.number_input("Loan Amount (₹)", min_value=100000, step=10000, value=500000)
            if loan_amount > annual_revenue:
                st.markdown("<p class='error-msg'>Loan amount cannot be greater than annual revenue!</p>", unsafe_allow_html=True)
            gst_billing = st.slider("GST Compliance (%)", min_value=50, max_value=100, step=1, value=80)

    with sub_col2:
        with st.container():
            credit_utilization_manual = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, step=1.0, value=30.0)
            credit_age = st.number_input("Credit Age (Years)", min_value=1, max_value=30, step=1, value=10)
            repayment_history = st.radio("Repayment History", [1, 0], horizontal=True)
    
    # Button in the input column, centered
    if st.button(" Generate CIBIL Score"):
        if loan_amount > annual_revenue:
            st.error("Fix the input errors before proceeding.")
        else:
            features = np.array([
                annual_revenue, loan_amount, gst_billing,
                repayment_history, credit_utilization_manual, credit_age
            ]).reshape(1, -1)

            predicted_score = rf_model.predict(features)[0]
            predicted_score = np.clip(predicted_score, 300, 900)
            
            # Store the results in session state so we can display them in the gauge column
            st.session_state.predicted_score = predicted_score
            st.session_state.show_gauge = True
            
            # Add this prediction to history
            new_prediction = pd.DataFrame({
                'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'annual_revenue': [annual_revenue],
                'loan_amount': [loan_amount],
                'gst_billing': [gst_billing],
                'credit_utilization': [credit_utilization_manual],
                'credit_age': [credit_age],
                'repayment_history': [repayment_history],
                'predicted_score': [predicted_score]
            })
            
            st.session_state.history = pd.concat([st.session_state.history, new_prediction], ignore_index=True)
            st.markdown("""<br><br>""", unsafe_allow_html=True)


# Function to generate gauge chart
def plot_cibil_gauge(score):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})

    # Define score ranges, colors, and labels
    sections = [300, 580, 670, 740, 800, 900]
    colors = ["darkgreen", "lightgreen", "yellow", "orange", "red"]  # Reversed order
    labels = ["Excellent", "Very Good", "Good", "Avg", "Below Avg"]  # Reversed order

    # Convert score to angle (300° = left, 900° = right)
    angle = np.interp(score, [300, 900], [np.pi, 0])

    # Plot colored segments with gaps
    for i in range(len(sections) - 1):
        start_angle = np.pi * (i / 5)
        end_angle = np.pi * ((i + 1) / 5) - 0.05  # Small gap

        ax.bar(
            x=np.linspace(start_angle, end_angle, 10),
            height=0.3,
            width=np.pi / 5 - 0.05,
            color=colors[i],
            alpha=1.0,
            align='center'
        )

        # Add text labels
        mid_angle = (start_angle + end_angle) / 2
        ax.text(mid_angle, 0.35, labels[i], ha='center', va='center', fontsize=11, fontweight='bold', color="black")

    # Add arrow indicator
    ax.annotate("", xy=(angle, 0.25), xytext=(np.pi, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    fig.patch.set_alpha(0)  # Make figure background transparent
    ax.set_facecolor("none")  

    # Hide extra elements
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    plt.subplots_adjust(bottom=-0.3)

    return fig

# Gauge section in the right column (75%)
with gauge_col:
    # Center the gauge display in this column
    gauge_space1, gauge_center, gauge_space2 = st.columns([1, 8, 1])
    
    with gauge_center:
        # Check if we have a prediction to show
        if 'show_gauge' in st.session_state and st.session_state.show_gauge:
            # Show the score
            st.markdown(
                f"""
                <div style="text-align: center; font-size: 32px; color: black;">
                Predicted CIBIL Score: {st.session_state.predicted_score:.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Display gauge chart
            fig = plot_cibil_gauge(st.session_state.predicted_score)
            st.pyplot(fig)

# Add space between gauge and dashboard
# st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

# Custom CSS for Card-like Styling
card_style = """
    <style>
        .card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-size: 16px;
            color:black;
        }
        .card h2 {
            color: #007bff;
        }
    </style>
"""

# ===== DASHBOARD SECTION =====
st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown(
#     """
#     <h2 style="text-align: center; color: #00008b; font-size: 36px; font-weight: bold;">
#         Credit Score Dashboard
#     </h2>
#     """,
#     unsafe_allow_html=True,
# )

st.markdown(
    """
    <div style="background-color: white; padding: 20px; border-radius: 10px;">
        <h1 style="text-align: center; color: #00008b; font-size: 52px; font-weight: bold;">
            Credit Score Dashboard
        </h1>
    </div>
    <br>
    <hr>
    """,
    unsafe_allow_html=True,
)


# st.markdown("<hr>", unsafe_allow_html=True)
# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["Score Analysis", "Historical Data", "Recommendations"])

# Tab 1: Score Analysis
with tab1:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    
    # Create columns for metrics
    if 'predicted_score' in st.session_state:
        # Create score breakdown metrics
        m1, m2, m3, m4 = st.columns(4)
        
        # Calculate score categories based on current input
        score = st.session_state.predicted_score
        
        # Define score category
        category = "Below Average"
        if score >= 580 and score < 670:
            category = " Average"
        elif score >= 670 and score < 740:
            category = "Good"
        elif score >= 740 and score < 800:
            category = "Very Good"
        elif score >= 800:
            category = "Excellent"
            
        # Calculate risk level
        risk_level = "High Risk"
        if score >= 580 and score < 670:
            risk_level = "Medium Risk"
        elif score >= 670 and score < 740:
            risk_level = "Low Risk"
        elif score >= 740:
            risk_level = "Very Low Risk"
            
        # Calculate approval chances
        approval_chance = "< 30%"
        if score >= 580 and score < 670:
            approval_chance = "30-60%"
        elif score >= 670 and score < 740:
            approval_chance = "60-80%"
        elif score >= 740:
            approval_chance = "> 80%"
            
        # Get loan interest impact
        interest_impact = "High Interest"
        if score >= 580 and score < 670:
            interest_impact = "Above Average"
        elif score >= 670 and score < 740:
            interest_impact = "Average"
        elif score >= 740:
            interest_impact = "Below Average"
        
        with m1:
            st.markdown(
                """
                <div class="dashboard-card">
                    <div class="metric-label">Score Category</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format(category),
                unsafe_allow_html=True
            )
            
        with m2:
            st.markdown(
                """
                <div class="dashboard-card">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format(risk_level),
                unsafe_allow_html=True
            )
            
        with m3:
            st.markdown(
                """
                <div class="dashboard-card">
                    <div class="metric-label">Approval Chance</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format(approval_chance),
                unsafe_allow_html=True
            )
            
        with m4:
            st.markdown(
                """
                <div class="dashboard-card">
                    <div class="metric-label">Interest Rate Impact</div>
                    <div class="metric-value">{}</div>
                </div>
                """.format(interest_impact),
                unsafe_allow_html=True
            )
        
        # Factor Impact Analysis
        st.subheader("Factor Impact Analysis")
        
        # Create factor impact chart
        factor_impact = pd.DataFrame({
            'Factor': ['Annual Revenue', 'Loan Amount', 'GST Compliance', 'Credit Utilization', 'Credit Age', 'Repayment History'],
            'Impact': [0.2, -0.15, 0.18, -0.22, 0.12, 0.3]  # These values would ideally come from feature importance
        })
        
        # Create an impact chart
        impact_chart = alt.Chart(factor_impact).mark_bar().encode(
            x=alt.X('Impact', axis=alt.Axis(title='Impact on Credit Score')),
            y=alt.Y('Factor', sort='-x', axis=alt.Axis(title=None)),
            color=alt.condition(
                alt.datum.Impact > 0,
                alt.value('green'),
                alt.value('red')
            )
        ).properties(
            height=250
        )
        
        st.altair_chart(impact_chart, use_container_width=True)
        
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 50px; color: #666;">
                <h3>Generate a CIBIL score to see detailed analysis</h3>
                <p>Use the form above to input your financial details and click 'Generate CIBIL Score' to see your results.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Historical Data
with tab2:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    
    if not st.session_state.history.empty:
        # Show history table
        st.subheader("Previous Predictions")
        st.dataframe(st.session_state.history, hide_index=True)
        
        # Only show chart if we have more than one prediction
        if len(st.session_state.history) > 1:
            st.subheader("Score Trend")
            
            # Create a line chart of historical scores
            history_chart = alt.Chart(st.session_state.history).mark_line(point=True).encode(
                x='timestamp:T',
                y=alt.Y('predicted_score:Q', scale=alt.Scale(domain=[300, 900])),
                tooltip=['timestamp', 'predicted_score', 'annual_revenue', 'loan_amount']
            ).properties(
                height=300
            )
            
            st.altair_chart(history_chart, use_container_width=True)
            
            # Add a button to clear history
            if st.button("Clear History"):
                st.session_state.history = pd.DataFrame(columns=[
                    'timestamp', 'annual_revenue', 'loan_amount', 'gst_billing',
                    'credit_utilization', 'credit_age', 'repayment_history', 'predicted_score'
                ])
                st.experimental_rerun()
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 50px; color: #666;">
                <h3>No historical data available</h3>
                <p>Generate multiple CIBIL scores to track your progress over time.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tab 3: Recommendations
# with tab3:
#     st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    
#     if 'predicted_score' in st.session_state:
#         score = st.session_state.predicted_score
        
#         st.subheader("Personalized Recommendations")
        
#         # Different recommendations based on score
#         if score < 580:
#             st.markdown(
#                 """
#                 <div class="dashboard-card">
#                     <h3>Improvement Plan for Poor Score (300-579)</h3>
#                     <ul>
#                         <li><strong>Focus on payment history:</strong> Ensure all future payments are made on time.</li>
#                         <li><strong>Reduce credit utilization:</strong> Try to keep your credit utilization below 30%.</li>
#                         <li><strong>Consider a secured credit card:</strong> Build credit history with a secured card.</li>
#                         <li><strong>Check for errors:</strong> Review your credit report for inaccuracies.</li>
#                         <li><strong>Be patient:</strong> Improving from this range takes time, typically 12-18 months of consistent good behavior.</li>
#                     </ul>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#         elif score >= 580 and score < 670:
#             st.markdown(
#                 """
#                 <div class="dashboard-card">
#                     <h3>Improvement Plan for Below Average Score (580-669)</h3>
#                     <ul>
#                         <li><strong>Maintain perfect payment history:</strong> Late payments have a significant negative impact.</li>
#                         <li><strong>Reduce high balances:</strong> Pay down revolving credit to below 30% utilization.</li>
#                         <li><strong>Avoid new credit applications:</strong> Multiple inquiries can lower your score.</li>
#                         <li><strong>Increase GST compliance:</strong> Ensure your business tax filings are timely and accurate.</li>
#                         <li><strong>Target timeframe:</strong> With consistent positive actions, you can see improvements in 6-12 months.</li>
#                     </ul>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#         elif score >= 670 and score < 740:
#             st.markdown(
#                 """
#                 <div class="dashboard-card">
#                     <h3>Improvement Plan for Good Score (670-739)</h3>
#                     <ul>
#                         <li><strong>Optimize credit mix:</strong> Having different types of credit can improve your score.</li>
#                         <li><strong>Maintain low utilization:</strong> Keep credit card balances below 20% of limits.</li>
#                         <li><strong>Avoid closing old accounts:</strong> Longer credit history improves your score.</li>
#                         <li><strong>Increase annual revenue reporting:</strong> Higher business income can positively impact scoring.</li>
#                         <li><strong>Target timeframe:</strong> You can reach the very good range in 3-6 months with these optimizations.</li>
#                     </ul>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#         elif score >= 740:
#             st.markdown(
#                 """
#                 <div class="dashboard-card">
#                     <h3>Maintenance Plan for Excellent Score (740+)</h3>
#                     <ul>
#                         <li><strong>Maintain your excellent habits:</strong> Continue making payments on time and keeping utilization low.</li>
#                         <li><strong>Monitor your credit report:</strong> Check regularly for errors or fraudulent activity.</li>
#                         <li><strong>Apply for credit strategically:</strong> Only apply for new credit when truly needed.</li>
#                         <li><strong>Negotiate better terms:</strong> With your excellent score, you can ask for lower interest rates.</li>
#                         <li><strong>Consider a business expansion loan:</strong> Your score qualifies you for premium business financing options.</li>
#                     </ul>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
        
#         # Additional tips
#         st.subheader("General Tips")
#         general_tips = pd.DataFrame({
#             'Tip': [
#                 "Keep credit card balances low",
#                 "Pay bills on time",
#                 "Don't close old credit accounts",
#                 "Limit hard inquiries",
#                 "Report all business income"
#             ],
#             'Potential Impact': [30, 35, 15, 10, 20]
#         })
        
#         # Create a tip impact chart
#         tips_chart = alt.Chart(general_tips).mark_bar().encode(
#             x=alt.X('Potential Impact', axis=alt.Axis(title='Potential Score Impact')),
#             y=alt.Y('Tip', sort='-x', axis=alt.Axis(title=None)),
#             color=alt.Color('Potential Impact', scale=alt.Scale(scheme='blues'))
#         ).properties(
#             height=200
#         )
        
#         st.altair_chart(tips_chart, use_container_width=True)
        
#     else:
#         st.markdown(
#             """
#             <div style="text-align: center; padding: 50px; color: #666;">
#                 <h3>Generate a CIBIL score to see personalized recommendations</h3>
#                 <p>Use the form above to input your financial details and click 'Generate CIBIL Score'.</p>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )
    
#     st.markdown("</div>", unsafe_allow_html=True)


with tab3:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    
    if 'predicted_score' in st.session_state:
        score = st.session_state.predicted_score
        
        st.subheader("Personalized Recommendations")
        
        # Different recommendations based on score
        if score < 580:
            st.markdown(
                """
                <div class="dashboard-card" style="color: black;">
                    <h3>Improvement Plan for Poor Score (300-579)</h3>
                    <ul>
                        <li><strong>Focus on payment history:</strong> Ensure all future payments are made on time.</li>
                        <li><strong>Reduce credit utilization:</strong> Try to keep your credit utilization below 30%.</li>
                        <li><strong>Consider a secured credit card:</strong> Build credit history with a secured card.</li>
                        <li><strong>Check for errors:</strong> Review your credit report for inaccuracies.</li>
                        <li><strong>Be patient:</strong> Improving from this range takes time, typically 12-18 months of consistent good behavior.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif score >= 580 and score < 670:
            st.markdown(
                """
                <div class="dashboard-card" style="color: black;">
                    <h3>Improvement Plan for Below Average Score (580-669)</h3>
                    <ul>
                        <li><strong>Maintain perfect payment history:</strong> Late payments have a significant negative impact.</li>
                        <li><strong>Reduce high balances:</strong> Pay down revolving credit to below 30% utilization.</li>
                        <li><strong>Avoid new credit applications:</strong> Multiple inquiries can lower your score.</li>
                        <li><strong>Increase GST compliance:</strong> Ensure your business tax filings are timely and accurate.</li>
                        <li><strong>Target timeframe:</strong> With consistent positive actions, you can see improvements in 6-12 months.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif score >= 670 and score < 740:
            st.markdown(
                """
                <div class="dashboard-card" style="color: black;">
                    <h3>Improvement Plan for Good Score (670-739)</h3>
                    <ul>
                        <li><strong>Optimize credit mix:</strong> Having different types of credit can improve your score.</li>
                        <li><strong>Maintain low utilization:</strong> Keep credit card balances below 20% of limits.</li>
                        <li><strong>Avoid closing old accounts:</strong> Longer credit history improves your score.</li>
                        <li><strong>Increase annual revenue reporting:</strong> Higher business income can positively impact scoring.</li>
                        <li><strong>Target timeframe:</strong> You can reach the very good range in 3-6 months with these optimizations.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif score >= 740:
            st.markdown(
                """
                <div class="dashboard-card" style="color: black;">
                    <h3>Maintenance Plan for Excellent Score (740+)</h3>
                    <ul>
                        <li><strong>Maintain your excellent habits:</strong> Continue making payments on time and keeping utilization low.</li>
                        <li><strong>Monitor your credit report:</strong> Check regularly for errors or fraudulent activity.</li>
                        <li><strong>Apply for credit strategically:</strong> Only apply for new credit when truly needed.</li>
                        <li><strong>Negotiate better terms:</strong> With your excellent score, you can ask for lower interest rates.</li>
                        <li><strong>Consider a business expansion loan:</strong> Your score qualifies you for premium business financing options.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Additional tips
        st.subheader("General Tips")
        general_tips = pd.DataFrame({
            'Tip': [
                "Keep credit card balances low",
                "Pay bills on time",
                "Don't close old credit accounts",
                "Limit hard inquiries",
                "Report all business income"
            ],
            'Potential Impact': [30, 35, 15, 10, 20]
        })
        
        # Create a tip impact chart
        tips_chart = alt.Chart(general_tips).mark_bar().encode(
            x=alt.X('Potential Impact', axis=alt.Axis(title='Potential Score Impact')),
            y=alt.Y('Tip', sort='-x', axis=alt.Axis(title=None)),
            color=alt.Color('Potential Impact', scale=alt.Scale(scheme='blues'))
        ).properties(
            height=200
        )
        
        st.altair_chart(tips_chart, use_container_width=True)
        
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 50px; color: #666;">
                <h3>Generate a CIBIL score to see personalized recommendations</h3>
                <p>Use the form above to input your financial details and click 'Generate CIBIL Score'.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(card_style, unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown(
#     """
#     <h2 style="text-align: center; color: #00008b; font-size: 36px; font-weight: bold;">
#        ABOUT CIBIL SCORE
#     </h2>
#     """,
#     unsafe_allow_html=True,
# )

st.markdown(
    """
    <div style="background-color: white; padding: 20px; border-radius: 10px;">
        <h1 style="text-align: center; color: #00008b; font-size: 52px; font-weight: bold;">
             ABOUT CIBIL SCORE
        </h1>
    </div>
    <br>
    <hr>
    """,
    unsafe_allow_html=True,
)

# Create columns for info cards
col1, col2, col3 = st.columns(3)

# Card 1: What is CIBIL Score?
with col1:
    st.markdown(
        """
        <div class='card'>
           <div class='title' style='font-size: 20px; font-weight: bold;'>What is a CIBIL Score?</div>
            <div class='content'>
                The CIBIL score is a three-digit number (300-900) that <br> represents creditworthiness. <br>
                A higher score improves loan approval chances.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Card 2: Importance of CIBIL Score
with col2:
    st.markdown(
        """
        <div class='card'>
         <div class='title' style='font-size: 20px; font-weight: bold;'>Why is it Important?</div>
            <div class='content'>
                Banks & lenders use the CIBIL score<br> to assess loan eligibility. <br>
                A score above 750 is considered excellent for approvals.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Card 3: Factors Affecting CIBIL Score
with col3:
    st.markdown(
        """
        <div class='card'>
            <div class='title' style='font-size: 20px; font-weight: bold;'>Factors Affecting Your Score</div>
            <div class='content'>
                - High Credit Utilization (above 50%) can lower your score. <br>
                - Timely Payments significantly impact your score <br>
                - Too Many Loan Applications can be a red flag for lenders<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
