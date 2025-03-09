# import streamlit as st
# import pandas as pd
# import random

# # Sample function to generate a random CIBIL Score
# def generate_cibil_score():
#     return random.randint(300, 900)

# # Function to determine risk category
# def get_risk_category(score):
#     if score >= 750:
#         return "Low Risk", "🟢"
#     elif score >= 600:
#         return "Medium Risk", "🟡"
#     else:
#         return "High Risk", "🔴"

# # UI Title
# st.set_page_config(page_title="CIBIL Score Report", layout="centered")
# st.title("📊CIBIL Score Report")

# # Upload financial data
# uploaded_file = st.file_uploader("Upload Business Financial Data (CSV)", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.success("File uploaded successfully! Processing data...")
    
#     # Display a sample of the uploaded data
#     st.subheader("📂 Uploaded Financial Data Sample")
#     st.dataframe(df.head())

#     # Generate random CIBIL Score for demonstration
#     cibil_score = generate_cibil_score()
#     risk_category, risk_icon = get_risk_category(cibil_score)

#     # Display CIBIL Score Report Card
#     st.subheader("🏦 Business Credit Score Report")
    
#     st.markdown(f"""
#         <div style="border-radius: 10px; padding: 20px; background-color: #f8f9fa; text-align: center;">
#             <h2 style="color: #007bff;">CIBIL Score: {cibil_score}</h2>
#             <h3>{risk_icon} {risk_category}</h3>
#         </div>
#     """, unsafe_allow_html=True)

#     # Loan Default Risk Prediction
#     default_risk = round(random.uniform(0.01, 0.5) * 100, 2)
#     st.subheader("⚠️ Loan Default Risk Prediction")
#     st.metric(label="Predicted Default Risk (%)", value=f"{default_risk}%", delta="-5% vs industry average")

#     # Financial Summary Section
#     st.subheader("📈 Financial Summary")
#     st.write("""
#     - **Annual Revenue:** ₹5,00,00,000  
#     - **Loan Amount Requested:** ₹50,00,000  
#     - **GST Compliance:** 95% ✅  
#     - **Past Defaults:** None  
#     - **Market Trend:** Growing 📈  
#     """)

#     # Explainability Section
#     st.subheader("🔍 Key Factors Influencing Score")
#     st.write("""
#     - ✅ High Annual Revenue & Positive Cash Flow  
#     - ✅ Strong GST Compliance & Bank Transactions  
#     - ⚠️ Moderate Loan Amount vs Revenue Ratio  
#     - ❌ Slightly Unstable Supplier Payments  
#     """)

#     # Download Report
#     st.subheader("📄 Download CIBIL Report")
#     st.download_button("Download Report (PDF)", "CIBIL_Report_Sample.pdf", "application/pdf")

# else:
#     st.warning("Please upload business financial data to generate the CIBIL score report.")

# # Footer
# st.markdown("---")
# st.markdown("🔗 Powered by AI | Built with ❤️ using Streamlit")


# import streamlit as st
# import numpy as np
# from cibil_model import predict_cibil  # Import trained model
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="CIBIL Score Predictor", layout="centered")
# st.title("📊 CIBIL Score Predictor")

# st.subheader("🔹 Enter Financial Details")

# # Annual Revenue (₹)
# annual_revenue_slider = st.slider("Annual Revenue (₹)", min_value=500000, max_value=50000000, step=10000)
# annual_revenue_manual = st.number_input("Or enter manually:", min_value=500000, step=10000, value=annual_revenue_slider)

# # Loan Amount (₹)
# loan_amount_slider = st.slider("Loan Amount (₹)", min_value=100000, max_value=5000000, step=10000)
# loan_amount_manual = st.number_input("Or enter manually:", min_value=100000, step=10000, value=loan_amount_slider)

# # GST Compliance (%)
# gst_billing_slider = st.slider("GST Compliance (%)", min_value=50, max_value=100, step=1)
# gst_billing_manual = st.number_input("Or enter manually:", min_value=50, max_value=100, step=1, value=gst_billing_slider)

# # Repayment History (0 or 1)
# repayment_history = st.radio("Repayment History", [0, 1])

# # Credit Utilization Ratio (0.0 - 1.0)
# # credit_utilization_slider = st.slider("Credit Utilization Ratio (0-1)", min_value=0.0, max_value=1.0, step=0.01)
# # credit_utilization_manual = st.number_input("Or enter manually:", min_value=0.0, max_value=1.0, step=0.01, value=credit_utilization_slider)

# # Credit Utilization Ratio (0 - 100)
# credit_utilization_slider = st.slider("Credit Utilization Ratio (0-100)", min_value=0.0, max_value=100.0, step=0.1)
# credit_utilization_manual = st.number_input("Or enter manually:", min_value=0.0, max_value=100.0, step=0.1, value=credit_utilization_slider)


# # Credit Age (Years)
# credit_age_slider = st.slider("Credit Age (Years)", min_value=1, max_value=30, step=1)
# credit_age_manual = st.number_input("Or enter manually:", min_value=1, max_value=30, step=1, value=credit_age_slider)

# # Button to generate CIBIL score
# if st.button("Generate CIBIL Score"):
#     user_features = [
#         annual_revenue_manual, loan_amount_manual, gst_billing_manual, repayment_history, credit_utilization_manual, credit_age_manual
#     ]
#     cibil_score = predict_cibil(user_features)
    
#     st.subheader("🏦 CIBIL Score Report")
#     st.metric(label="Predicted CIBIL Score", value=int(cibil_score))
#     import matplotlib.pyplot as plt
#     import numpy as np

# def plot_cibil_gauge(score):
#     fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})

#     # Define score ranges, colors, and labels
#     sections = [300, 580, 670, 740, 800, 900]
#     colors = ["red", "orange", "yellow", "lightgreen", "darkgreen"]
#     labels = ["Below Avg", "Avg", "Good", "Very Good", "Excellent"]

#     # Convert score to angle (300° = left, 900° = right)
#     angle = np.interp(score, [300, 900], [np.pi, 0])

#     # Plot colored segments with gaps
#     for i in range(len(sections) - 1):
#         start_angle = np.pi * (i / 5)
#         end_angle = np.pi * ((i + 1) / 5) - 0.05  # Small gap

#         ax.bar(
#             x=np.linspace(start_angle, end_angle, 10),
#             height=0.3,
#             width=np.pi / 5 - 0.05,  # Adjust width to create gaps
#             color=colors[i],
#             alpha=1.0,
#             align='center'
#         )

#         # Add text labels above segments
#         mid_angle = (start_angle + end_angle) / 2
#         ax.text(mid_angle, 0.35, labels[i], ha='center', va='center', fontsize=6, fontweight='bold', color="black")

#     # Add the score arrow (thin arrow now)
#     ax.annotate("", xy=(angle, 0.25), xytext=(np.pi, 0),
#                 arrowprops=dict(arrowstyle="->", color="black", lw=2))

#     # Hide extra elements
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.spines['polar'].set_visible(False)
#     plt.subplots_adjust(bottom=-0.3)  # Adjust bottom spacing
#     # Move title below the meter
#     # plt.figtext(0.5, 0, "CIBIL Score Gauge", ha="center", fontsize=12, fontweight="bold", color="black")

#     return fig

# if st.button("🚀 Generate CIBIL Score"):
#     user_features = [
#         annual_revenue_manual,
#         loan_amount_manual,
#         gst_billing_manual,
#         repayment_history,
#         credit_utilization_manual,
#         credit_age_manual
#     ]

#     cibil_score = predict_cibil(user_features)

#     # Display the Gauge Chart
#     st.subheader("📊 CIBIL Score Analysis")
#     fig = plot_cibil_gauge(cibil_score)
#     st.pyplot(fig)

#     # Display Score Card
#     st.markdown(
#         f"""
#         <div style="border-radius: 10px; padding: 20px; background-color: #fff; text-align: center; border: 2px solid black;">
#             <h2 style="color: black;">CIBIL Score: {cibil_score:.2f}</h2>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import altair as alt
import joblib


# Load trained model
from cibil_model import rf_model

# Streamlit page setup
st.set_page_config(page_title="<b>CIBIL Score Predictor", layout="wide")
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
            # background-image: url("https://icdproperty.com.au/wp-content/uploads/2016/08/blue-shape-bg-lrg.jpg");
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
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style="text-align: center; color: #00008b; font-size: 52px; font-weight: bold;">
        CIBIL Score Predictor
    </h1>
    <div> <br> </div>
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

  

# import streamlit as st

# Page Configurations
# st.set_page_config(page_title="CIBIL Score Insights", layout="wide")
st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

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
st.markdown(card_style, unsafe_allow_html=True)
# Create columns for a better layout
col1, col2, col3 = st.columns(3)


# Card 1: What is CIBIL Score?
with col1:
    st.markdown(
        """
        <div class='card'>
            <div class='title'><b>What is a CIBIL Score?</div>
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
            <div class='title'><b>Why is it Important?</div>
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
            <div class='title'><b>Factors Affecting Your Score</div>
            <div class='content'>
                - High Credit Utilization (above 50%) can lower your score. <br>
                - Timely Payments significantly impact your score <br>
                - Too Many Loan Applications can be a red flag for lenders<br>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <h2 style="text-align: center; color: #00008b; font-size: 36px; font-weight: bold;">
        Credit Score Dashboard
    </h2>
    """,
    unsafe_allow_html=True,
)

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
            category = "Average"
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
with tab3:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    
    if 'predicted_score' in st.session_state:
        score = st.session_state.predicted_score
        
        st.subheader("Personalized Recommendations")
        
        # Different recommendations based on score
        if score < 580:
            st.markdown(
        """
        <style>
            .dashboard-card {
                color: black !important;
            }
        </style>
        <div class="dashboard-card">
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
                <div class="dashboard-card">
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
                <div class="dashboard-card">
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
                <div class="dashboard-card">
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

# More insights
# st.subheader("Additional Insights")
# st.markdown(
#     """
#     - **High Credit Utilization** (above 50%) can lower your score.
#     - **Timely Payments** significantly impact your score.
#     - **Longer Credit History** contributes to a better score.
#     - **Too Many Loan Applications** can be a red flag for lenders.
#     """
# )



# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from cibil_model import predict_cibil  # Import trained model

# # Page configuration
# st.set_page_config(
#     page_title="CIBIL Score Predictor",
#     page_icon="📊",
#     layout="wide"
# )

# # Custom CSS to improve aesthetics
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #1E3A8A;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         font-weight: 600;
#         color: #1E3A8A;
#         margin-top: 2rem;
#     }
#     .slider-container {
#         background-color: #F3F4F6;
#         padding: 20px;
#         border-radius: 10px;
#         margin-bottom: 20px;
#     }
#     .stButton>button {
#         background-color: #1E3A8A;
#         color: white;
#         font-weight: bold;
#         border-radius: 5px;
#         padding: 0.5rem 2rem;
#         width: 100%;
#     }
#     .info-box {
#         background-color: #E0F2FE;
#         padding: 15px;
#         border-radius: 5px;
#         margin-bottom: 20px;
#     }
#     .indicator {
#         font-size: 1.5rem;
#         font-weight: 700;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Header
# st.markdown('<p class="main-header">📊 CIBIL Score Predictor</p>', unsafe_allow_html=True)

# # Layout with columns for better organization
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.markdown('<p class="sub-header">🔹 Enter Financial Details</p>', unsafe_allow_html=True)
    
#     # Explanation box
#     st.markdown("""
#     <div class="info-box">
#     This tool predicts your CIBIL score based on your financial information. 
#     Adjust the sliders to match your business profile for an accurate prediction.
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Create two columns for the form layout
#     left_col, right_col = st.columns(2)
    
#     with left_col:
#         st.markdown('<div class="slider-container">', unsafe_allow_html=True)
#         # Annual Revenue (₹)
#         st.markdown("**Annual Revenue (₹)**")
#         annual_revenue = st.slider(
#             "Annual Revenue", 
#             min_value=500000, 
#             max_value=50000000, 
#             step=100000,
#             value=1000000,
#             format="%d",
#             label_visibility="collapsed"
#         )
        
#         # GST Compliance (%)
#         st.markdown("**GST Compliance (%)**")
#         gst_billing = st.slider(
#             "GST Compliance", 
#             min_value=50, 
#             max_value=100, 
#             step=1,
#             value=80,
#             label_visibility="collapsed"
#         )
        
#         # Credit Utilization Ratio
#         st.markdown("**Credit Utilization Ratio (%)**")
#         credit_utilization = st.slider(
#             "Credit Utilization", 
#             min_value=0.0, 
#             max_value=100.0, 
#             step=1.0,
#             value=30.0,
#             label_visibility="collapsed"
#         )
#         st.markdown('</div>', unsafe_allow_html=True)
        
#     with right_col:
#         st.markdown('<div class="slider-container">', unsafe_allow_html=True)
#         # Loan Amount (₹)
#         st.markdown("**Loan Amount (₹)**")
#         loan_amount = st.slider(
#             "Loan Amount", 
#             min_value=100000, 
#             max_value=5000000, 
#             step=50000,
#             value=500000,
#             format="%d",
#             label_visibility="collapsed"
#         )
        
#         # Repayment History
#         st.markdown("**Repayment History**")
#         repayment_history = st.radio(
#             "Repayment History",
#             options=["Poor (0)", "Good (1)"],
#             index=1,
#             horizontal=True,
#             label_visibility="collapsed"
#         )
#         # Convert to numeric value
#         repayment_history = 1 if repayment_history == "Good (1)" else 0
        
#         # Credit Age (Years)
#         st.markdown("**Credit Age (Years)**")
#         credit_age = st.slider(
#             "Credit Age", 
#             min_value=1, 
#             max_value=30, 
#             step=1,
#             value=5,
#             label_visibility="collapsed"
#         )
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Generate button
#     generate_button = st.button("🚀 Generate CIBIL Score")

# # Function to create the gauge chart
# def plot_cibil_gauge(score):
#     # Map score ranges to color segments
#     score_ranges = {
#         (300, 579): {"color": "#FF4B4B", "label": "Poor"},
#         (580, 669): {"color": "#FFA500", "label": "Fair"},
#         (670, 739): {"color": "#FFD700", "label": "Good"},
#         (740, 799): {"color": "#90EE90", "label": "Very Good"},
#         (800, 900): {"color": "#32CD32", "label": "Excellent"}
#     }
    
#     # Get score category and color
#     score_category = None
#     score_color = None
#     for score_range, details in score_ranges.items():
#         if score_range[0] <= score <= score_range[1]:
#             score_category = details["label"]
#             score_color = details["color"]
#             break
    
#     fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})

#     # Define score ranges and angles
#     sections = [300, 580, 670, 740, 800, 900]
#     colors = ["#FF4B4B", "#FFA500", "#FFD700", "#90EE90", "#32CD32"]
#     labels = ["Poor", "Fair", "Good", "Very Good", "Excellent"]

#     # Convert score to angle (300° = left, 900° = right)
#     angle = np.interp(score, [300, 900], [np.pi, 0])

#     # Plot colored segments with gaps
#     for i in range(len(sections) - 1):
#         start_angle = np.pi * (i / 5)
#         end_angle = np.pi * ((i + 1) / 5) - 0.05  # Small gap

#         ax.bar(
#             x=np.linspace(start_angle, end_angle, 10),
#             height=0.3,
#             width=np.pi / 5 - 0.05,  # Adjust width to create gaps
#             color=colors[i],
#             alpha=0.8,
#             align='center'
#         )

#         # Add text labels above segments
#         mid_angle = (start_angle + end_angle) / 2
#         ax.text(mid_angle, 0.38, labels[i], ha='center', va='center', fontsize=9, fontweight='bold', color="black")

#     # Add score value at the bottom
#     ax.text(np.pi/2, -0.15, f"{int(score)}", ha='center', va='center', fontsize=18, fontweight='bold', color=score_color)
#     ax.text(np.pi/2, -0.25, f"{score_category}", ha='center', va='center', fontsize=12, color=score_color)

#     # Add the score arrow
#     ax.annotate("", xy=(angle, 0.25), xytext=(angle, 0),
#                 arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

#     # Hide extra elements
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.spines['polar'].set_visible(False)
    
#     # Set min/max values on the gauge
#     ax.text(np.pi, -0.05, "300", ha='center', va='center', fontsize=8, color="dimgray")
#     ax.text(0, -0.05, "900", ha='center', va='center', fontsize=8, color="dimgray")

#     plt.tight_layout()
#     return fig, score_category, score_color

# # Display results when button is clicked
# if generate_button:
#     user_features = [
#         annual_revenue,
#         loan_amount,
#         gst_billing,
#         repayment_history,
#         credit_utilization,
#         credit_age
#     ]

#     # Predict CIBIL score
#     cibil_score = predict_cibil(user_features)
    
#     # Create gauge and get score category
#     fig, score_category, score_color = plot_cibil_gauge(cibil_score)
    
#     with col2:
#         st.markdown('<p class="sub-header">📈 Your Score Results</p>', unsafe_allow_html=True)
        
#         # Display the score card
#         st.markdown(
#             f"""
#             <div style="border-radius: 10px; padding: 15px; background-color: {score_color}22; 
#                  text-align: center; border: 2px solid {score_color}; margin-bottom: 20px;">
#                 <h2 style="color: {score_color}; margin: 0;">CIBIL Score</h2>
#                 <p class="indicator" style="color: {score_color}; font-size: 2.5rem; margin: 10px 0;">{int(cibil_score)}</p>
#                 <p style="font-weight: bold; margin: 0;">{score_category}</p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
        
#         # Display the gauge chart
#         st.pyplot(fig)
        
#         # Score interpretation
#         st.markdown(
#             f"""
#             <div style="border-radius: 10px; padding: 15px; background-color: #F3F4F6; margin-top: 20px;">
#                 <h3 style="color: #1E3A8A; margin-top: 0;">Score Interpretation</h3>
#                 <p><b>300-579:</b> Poor credit standing, difficult to get loans</p>
#                 <p><b>580-669:</b> Fair credit, higher interest rates likely</p>
#                 <p><b>670-739:</b> Good credit, reasonable loan terms</p>
#                 <p><b>740-799:</b> Very good credit, favorable terms</p>
#                 <p><b>800-900:</b> Excellent credit, best rates and options</p>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )



# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# from cibil_model import predict_cibil  # Import trained model

# # Page configuration
# st.set_page_config(
#     page_title="CIBIL Score Predictor",
#     page_icon="📊",
#     layout="wide"
# )

# # Custom CSS to improve aesthetics
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #1E3A8A;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         font-weight: 600;
#         color: #1E3A8A;
#         margin-top: 2rem;
#     }
#     .slider-container {
#         background-color: #F3F4F6;
#         padding: 20px;
#         border-radius: 10px;
#         margin-bottom: 20px;
#     }
#     .stButton>button {
#         background-color: #1E3A8A;
#         color: white;
#         font-weight: bold;
#         border-radius: 5px;
#         padding: 0.5rem 2rem;
#         width: 100%;
#     }
#     .info-box {
#         background-color: #E0F2FE;
#         padding: 15px;
#         border-radius: 5px;
#         margin-bottom: 20px;
#     }
#     .indicator {
#         font-size: 1.5rem;
#         font-weight: 700;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Header
# st.markdown('<p class="main-header">📊 CIBIL Score Predictor</p>', unsafe_allow_html=True)

# # Layout with columns for better organization
# col1, col2 = st.columns([2, 1])

# # Function to create the gauge chart
# def plot_cibil_gauge(score):
#     # Map score ranges to color segments
#     score_ranges = {
#         (300, 579): {"color": "#FF4B4B", "label": "Poor"},
#         (580, 669): {"color": "#FFA500", "label": "Fair"},
#         (670, 739): {"color": "#FFD700", "label": "Good"},
#         (740, 799): {"color": "#90EE90", "label": "Very Good"},
#         (800, 900): {"color": "#32CD32", "label": "Excellent"}
#     }
    
#     # Get score category and color
#     score_category = None
#     score_color = None
#     for score_range, details in score_ranges.items():
#         if score_range[0] <= score <= score_range[1]:
#             score_category = details["label"]
#             score_color = details["color"]
#             break
    
#     fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})

#     # Define score ranges and angles
#     sections = [300, 580, 670, 740, 800, 900]
#     colors = ["#FF4B4B", "#FFA500", "#FFD700", "#90EE90", "#32CD32"]
#     labels = ["Poor", "Fair", "Good", "Very Good", "Excellent"]

#     # Convert score to angle (300° = left, 900° = right)
#     angle = np.interp(score, [300, 900], [np.pi, 0])

#     # Plot colored segments with gaps
#     for i in range(len(sections) - 1):
#         start_angle = np.pi * (i / 5)
#         end_angle = np.pi * ((i + 1) / 5) - 0.05  # Small gap

#         ax.bar(
#             x=np.linspace(start_angle, end_angle, 10),
#             height=0.3,
#             width=np.pi / 5 - 0.05,  # Adjust width to create gaps
#             color=colors[i],
#             alpha=0.8,
#             align='center'
#         )

#         # Add text labels above segments
#         mid_angle = (start_angle + end_angle) / 2
#         ax.text(mid_angle, 0.38, labels[i], ha='center', va='center', fontsize=9, fontweight='bold', color="black")

#     # Add score value at the bottom
#     ax.text(np.pi/2, -0.15, f"{int(score)}", ha='center', va='center', fontsize=18, fontweight='bold', color=score_color)
#     ax.text(np.pi/2, -0.25, f"{score_category}", ha='center', va='center', fontsize=12, color=score_color)

#     # Add the score arrow
#     ax.annotate("", xy=(angle, 0.25), xytext=(angle, 0),
#                 arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

#     # Hide extra elements
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.spines['polar'].set_visible(False)
    
#     # Set min/max values on the gauge
#     ax.text(np.pi, -0.05, "300", ha='center', va='center', fontsize=8, color="dimgray")
#     ax.text(0, -0.05, "900", ha='center', va='center', fontsize=8, color="dimgray")

#     plt.tight_layout()
#     return fig, score_category, score_color

# with col1:
#     st.markdown('<p class="sub-header">🔹 Enter Financial Details</p>', unsafe_allow_html=True)
    
#     # Explanation box
#     st.markdown("""
#     <div class="info-box">
#     This tool predicts your CIBIL score in real-time as you adjust the sliders.
#     Modify any parameter to see how it affects your overall score.
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Create two columns for the form layout
#     left_col, right_col = st.columns(2)
    
#     with left_col:
#         st.markdown('<div class="slider-container">', unsafe_allow_html=True)
#         # Annual Revenue (₹)
#         st.markdown("**Annual Revenue (₹)**")
#         annual_revenue = st.slider(
#             "Annual Revenue", 
#             min_value=500000, 
#             max_value=50000000, 
#             step=100000,
#             value=1000000,
#             format="%d",
#             label_visibility="collapsed",
#             key="annual_revenue"
#         )
        
#         # GST Compliance (%)
#         st.markdown("**GST Compliance (%)**")
#         gst_billing = st.slider(
#             "GST Compliance", 
#             min_value=50, 
#             max_value=100, 
#             step=1,
#             value=80,
#             label_visibility="collapsed",
#             key="gst_billing"
#         )
        
#         # Credit Utilization Ratio
#         st.markdown("**Credit Utilization Ratio (%)**")
#         credit_utilization = st.slider(
#             "Credit Utilization", 
#             min_value=0.0, 
#             max_value=100.0, 
#             step=1.0,
#             value=30.0,
#             label_visibility="collapsed",
#             key="credit_utilization"
#         )
#         st.markdown('</div>', unsafe_allow_html=True)
        
#     with right_col:
#         st.markdown('<div class="slider-container">', unsafe_allow_html=True)
#         # Loan Amount (₹)
#         st.markdown("**Loan Amount (₹)**")
#         loan_amount = st.slider(
#             "Loan Amount", 
#             min_value=100000, 
#             max_value=5000000, 
#             step=50000,
#             value=500000,
#             format="%d",
#             label_visibility="collapsed",
#             key="loan_amount"
#         )
        
#         # Repayment History
#         st.markdown("**Repayment History**")
#         repayment_history = st.radio(
#             "Repayment History",
#             options=["Poor (0)", "Good (1)"],
#             index=1,
#             horizontal=True,
#             label_visibility="collapsed",
#             key="repayment_history"
#         )
#         # Convert to numeric value
#         repayment_history = 1 if repayment_history == "Good (1)" else 0
        
#         # Credit Age (Years)
#         st.markdown("**Credit Age (Years)**")
#         credit_age = st.slider(
#             "Credit Age", 
#             min_value=1, 
#             max_value=30, 
#             step=1,
#             value=5,
#             label_visibility="collapsed",
#             key="credit_age"
#         )
#         st.markdown('</div>', unsafe_allow_html=True)

# # Calculate the score in real-time based on current slider values
# user_features = [
#     annual_revenue,
#     loan_amount,
#     gst_billing,
#     repayment_history,
#     credit_utilization,
#     credit_age
# ]

# # Predict CIBIL score
# cibil_score = predict_cibil(user_features)

# # Create gauge and get score category
# fig, score_category, score_color = plot_cibil_gauge(cibil_score)

# # Display the results in the right column
# with col2:
#     st.markdown('<p class="sub-header">📈 Live Score Results</p>', unsafe_allow_html=True)
    
#     # Display the score card
#     st.markdown(
#         f"""
#         <div style="border-radius: 10px; padding: 15px; background-color: {score_color}22; 
#              text-align: center; border: 2px solid {score_color}; margin-bottom: 20px;">
#             <h2 style="color: {score_color}; margin: 0;">CIBIL Score</h2>
#             <p class="indicator" style="color: {score_color}; font-size: 2.5rem; margin: 10px 0;">{int(cibil_score)}</p>
#             <p style="font-weight: bold; margin: 0;">{score_category}</p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
    
#     # Display the gauge chart
#     st.pyplot(fig)
    
#     # Score interpretation
#     st.markdown(
#         f"""
#         <div style="border-radius: 10px; padding: 15px; background-color: #F3F4F6; margin-top: 20px;">
#             <h3 style="color: #1E3A8A; margin-top: 0;">Score Interpretation</h3>
#             <p><b>300-579:</b> Poor credit standing, difficult to get loans</p>
#             <p><b>580-669:</b> Fair credit, higher interest rates likely</p>
#             <p><b>670-739:</b> Good credit, reasonable loan terms</p>
#             <p><b>740-799:</b> Very good credit, favorable terms</p>
#             <p><b>800-900:</b> Excellent credit, best rates and options</p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     # Add a what-if analysis section
#     st.markdown(
#         f"""
#         <div style="border-radius: 10px; padding: 15px; background-color: #E0F2FE; margin-top: 20px;">
#             <h3 style="color: #1E3A8A; margin-top: 0;">Adjustment Impact</h3>
#             <p>Try adjusting the sliders to see how different factors affect your CIBIL score:</p>
#             <ul>
#                 <li>Increasing <b>GST Compliance</b> generally improves your score</li>
#                 <li>Lower <b>Credit Utilization</b> (below 30%) often leads to better scores</li>
#                 <li>Longer <b>Credit Age</b> demonstrates stability</li>
#                 <li>Good <b>Repayment History</b> is critical for high scores</li>
#             </ul>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )