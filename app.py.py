import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import yfinance as yf
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from openai import OpenAI
import warnings
import base64

# Suppress unnecessary future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration and Initialization ---
st.set_page_config(
    page_title="Finance Co-Pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

USER_DATA_FILE = "user_data.json"

# --- Caching for expensive data calls ---
@st.cache_data
def get_stock_data(tickers, period="1y"):
    """Cached function to download stock data from yfinance."""
    return yf.download(tickers, period=period, progress=False)

@st.cache_data
def get_stock_news(ticker):
    """Cached function to get news for a specific stock."""
    try:
        return yf.Ticker(ticker).news
    except Exception:
        return []

# --- Ollama AI Integration ---
def get_ollama_response(prompt: str) -> str:
    """Generates a response from the local Ollama API."""
    try:
        client = OpenAI(
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",  # required by the library but not used by Ollama
        )
        completion = client.chat.completions.create(
            model="jjansen/adapt-finance-llama2-7b", # The model pulled during setup
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        if "Connection refused" in str(e):
            return "Error: Could not connect to the Ollama server. Please ensure it is running in the background."
        return f"An error occurred with the Ollama API call: {e}"

# --- Data Persistence ---
def save_data():
    """Saves user data from session_state to a JSON file."""
    if "user_data" in st.session_state:
        data_to_save = st.session_state.user_data.copy()
        # Convert DataFrames to JSON-serializable format
        for key in ['transactions', 'investments']:
            if key in data_to_save and isinstance(data_to_save[key], pd.DataFrame):
                df_copy = data_to_save[key].copy()
                if 'Date' in df_copy.columns:
                    # Ensure Date column is string before saving
                    df_copy['Date'] = pd.to_datetime(df_copy['Date']).dt.isoformat()
                data_to_save[key] = df_copy.to_dict('records')

        with open(USER_DATA_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)

def load_data(uploaded_file=None):
    """Loads user data into session_state from a file or uploaded data."""
    data_source = None
    if uploaded_file:
        data_source = json.load(uploaded_file)
    else:
        try:
            with open(USER_DATA_FILE, 'r') as f:
                data_source = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st.session_state.user_data = get_default_data()
            return

    st.session_state.user_data = get_default_data()
    if data_source:
        st.session_state.user_data.update(data_source)

    # Convert transaction and investment records back to DataFrames
    for key in ['transactions', 'investments']:
        df = pd.DataFrame(st.session_state.user_data.get(key, []))
        if 'Date' in df.columns and not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
        st.session_state.user_data[key] = df

def get_default_data():
    """Returns a default data structure for a new user."""
    return {
        "profile": {"name": None, "monthly_income": 5000},
        "transactions": pd.DataFrame(columns=["Date", "Description", "Category", "Amount"]),
        "investments": pd.DataFrame(columns=["Ticker", "Shares", "Avg_Cost"]),
        "budgets": {
            "Groceries": 500, "Dining Out": 200, "Shopping": 150,
            "Utilities": 150, "Transportation": 100, "Entertainment": 100, "Other": 100
        },
        "goals": [],
        "subscriptions": [],
        "debts": [],
    }

# --- UI Pages ---
def data_setup_page():
    st.title("👋 Welcome to Your Finance Co-Pilot!")
    st.write("This is a powerful, all-in-one application to manage your entire financial life. Let's start by setting up your profile.")
    with st.form("setup_form"):
        name = st.text_input("What should I call you?")
        income = st.number_input("Your Net Monthly Income ($)", min_value=0, value=5000)
        initial_balance = st.number_input("Your Current Checking/Savings Balance ($)", min_value=0.0, value=1000.0, step=100.0)
        if st.form_submit_button("Save and Begin"):
            if not name:
                st.error("Please enter your name.")
            else:
                st.session_state.user_data['profile'] = {"name": name, "monthly_income": income}
                initial_transaction = pd.DataFrame([{"Date": pd.to_datetime(datetime.now()), "Description": "Initial Balance", "Category": "Income", "Amount": initial_balance}])
                st.session_state.user_data['transactions'] = initial_transaction
                save_data()
                st.success("Your profile is set up! Welcome aboard.")
                st.rerun()

def dashboard_page():
    profile = st.session_state.user_data['profile']
    st.title(f"📊 Dashboard for {profile['name']}")

    trans = st.session_state.user_data.get('transactions', pd.DataFrame())
    if trans.empty:
        st.info("No transaction data yet. Add one to see your dashboard.")
        return

    current_balance = trans['Amount'].sum()
    monthly_income = profile['monthly_income']
    now = datetime.now()
    current_month_trans = trans[(trans['Date'].dt.month == now.month) & (trans['Date'].dt.year == now.year)]
    monthly_expenses = -current_month_trans[current_month_trans['Amount'] < 0]['Amount'].sum()
    monthly_savings = monthly_income - monthly_expenses

    # --- Financial Health Score ---
    savings_rate = (monthly_savings / monthly_income * 100) if monthly_income > 0 else 0
    score = min(max(savings_rate, 0), 100)
    score_color = "green" if score >= 20 else "orange" if score >= 10 else "red"

    # --- UI Layout ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Balance", f"${current_balance:,.2f}")
    col2.metric("Monthly Expenses", f"${monthly_expenses:,.2f}", delta_color="inverse")
    col3.metric("Monthly Net", f"${monthly_savings:,.2f}", help="Income - Expenses")
    with col4:
        st.markdown(f"**Financial Health**")
        st.markdown(f"<p style='color:{score_color}; font-size: 24px; font-weight: bold;'>{score:.0f}/100</p>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("💰 Cash Flow Over Time")
        trans_sorted = trans.sort_values(by="Date")
        trans_sorted['Cumulative'] = trans_sorted['Amount'].cumsum()
        fig = px.line(trans_sorted, x='Date', y='Cumulative', title="Account Balance History", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 This Month's Spending")
        expense_df = current_month_trans[current_month_trans['Amount'] < 0].copy()
        expense_df['Amount'] = expense_df['Amount'].abs()
        if not expense_df.empty:
            fig_pie = px.pie(expense_df, values='Amount', names='Category', hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write("No expenses recorded this month.")

    st.markdown("---")

    st.subheader("Recent Transactions")
    st.dataframe(trans.sort_values(by="Date", ascending=False).head(10), use_container_width=True)

def finance_chatbot_page():
    st.title("🤖 AI Co-Pilot Chat")
    st.markdown("Your personal AI assistant for complex financial questions, powered by your local Ollama model. It has context on your financial situation to provide personalized advice.")

    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Ask about tax strategies, investment ideas, or budget optimization..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Co-Pilot is thinking..."):
                # --- Create Financial Context ---
                profile = st.session_state.user_data['profile']
                trans = st.session_state.user_data.get('transactions', pd.DataFrame())
                investments = st.session_state.user_data.get('investments', pd.DataFrame())
                budgets = st.session_state.user_data.get('budgets', {})
                goals = st.session_state.user_data.get('goals', [])

                context = f"You are an expert financial analyst providing advice to {profile['name']}. " \
                          f"Their monthly income is ${profile['monthly_income']}. "
                if not trans.empty:
                    current_balance = trans['Amount'].sum()
                    monthly_expenses = -trans[trans['Date'].dt.month == datetime.now().month]['Amount'][trans['Amount']<0].sum()
                    context += f"Their current account balance is ${current_balance:.2f}. Their expenses this month are ${monthly_expenses:.2f}. "
                if budgets:
                    context += f"Their monthly budget is as follows: {json.dumps(budgets)}. "
                if not investments.empty:
                    context += f"They have an investment portfolio with these holdings: {investments.to_string()}. "
                if goals:
                    context += f"Their financial goals are: {goals}. "

                full_prompt = f"{context}Based on this comprehensive financial overview, answer the following question concisely and helpfully: {prompt}"

                assistant_response = get_ollama_response(full_prompt)
                placeholder.markdown(assistant_response)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

def investment_explorer_page():
    st.title("📈 Investment Explorer")

    inv_df = st.session_state.user_data.get('investments', pd.DataFrame()).copy()

    if not inv_df.empty:
        tickers = inv_df["Ticker"].tolist()
        try:
            # Fetch current prices
            data_today = yf.download(tickers, period="1d", progress=False)
            if not data_today.empty:
                if len(tickers) == 1:
                    last_close = data_today['Close'].iloc[-1]
                    inv_df['Current_Price'] = last_close
                else:
                    last_close = data_today['Close'].iloc[-1]
                    inv_df['Current_Price'] = inv_df['Ticker'].map(last_close)

                inv_df['Market_Value'] = inv_df['Shares'] * inv_df['Current_Price']
                inv_df['Total_Cost'] = inv_df['Shares'] * inv_df['Avg_Cost']
                inv_df['Gain_Loss'] = inv_df['Market_Value'] - inv_df['Total_Cost']

                total_value = inv_df['Market_Value'].sum()
                total_cost = inv_df['Total_Cost'].sum()
                total_gain_loss = inv_df['Gain_Loss'].sum()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Portfolio Value", f"${total_value:,.2f}")
                col2.metric("Total Cost Basis", f"${total_cost:,.2f}")
                col3.metric("Total Unrealized P/L", f"${total_gain_loss:,.2f}", f"{((total_gain_loss/total_cost)*100 if total_cost else 0):.2f}%")

                st.subheader("Your Holdings")
                st.dataframe(inv_df.style.format({
                    'Avg_Cost': '${:,.2f}', 'Current_Price': '${:,.2f}',
                    'Market_Value': '${:,.2f}', 'Total_Cost': '${:,.2f}', 'Gain_Loss': '${:,.2f}'
                }), use_container_width=True)
            else:
                 st.error(f"Could not fetch data for tickers: {tickers}. Please check symbols.")
        except Exception as e:
            st.error(f"Could not fetch live market data: {e}")
            st.dataframe(inv_df, use_container_width=True)

        st.markdown("---")
        st.header("🔬 Portfolio Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Diversification")
            fig = px.pie(inv_df, values='Market_Value', names='Ticker', title='Portfolio Allocation by Market Value', hole=0.3)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Live Market News")
            selected_ticker = st.selectbox("Select a stock to see news", tickers)
            if selected_ticker:
                news = get_stock_news(selected_ticker)
                if news:
                    for item in news[:5]:
                        st.markdown(f"**[{item['title']}]({item['link']})** - *{item['publisher']}*")
                else:
                    st.write("No recent news found for this ticker.")
    else:
        st.info("No investments added yet. Add a holding below to get started.")

    with st.expander("💹 Add or Update a Holding"):
        with st.form("investment_form", clear_on_submit=True):
            cols = st.columns(3)
            ticker = cols[0].text_input("Stock Ticker (e.g., AAPL)").upper()
            shares = cols[1].number_input("Number of Shares", min_value=0.0001, format="%.4f")
            avg_cost = cols[2].number_input("Average Cost Per Share ($)", min_value=0.01, format="%.2f")

            if st.form_submit_button("Add/Update Holding"):
                if not ticker or shares <= 0 or avg_cost <= 0:
                    st.warning("Please enter valid details for the holding.")
                else:
                    inv_df_current = st.session_state.user_data.get('investments', pd.DataFrame())
                    if ticker in inv_df_current['Ticker'].values:
                        inv_df_current.loc[inv_df_current['Ticker'] == ticker, ['Shares', 'Avg_Cost']] = [shares, avg_cost]
                    else:
                        new_inv = pd.DataFrame([{"Ticker": ticker, "Shares": shares, "Avg_Cost": avg_cost}])
                        inv_df_current = pd.concat([inv_df_current, new_inv], ignore_index=True)
                    st.session_state.user_data['investments'] = inv_df_current
                    save_data()
                    st.success(f"Holding for {ticker} updated.")
                    st.rerun()

def budgeting_page():
    st.title("💰 Budgeting")
    st.markdown("Track your monthly spending against your defined budget.")

    budgets = st.session_state.user_data.get('budgets', {})
    trans = st.session_state.user_data.get('transactions', pd.DataFrame())

    now = datetime.now()
    current_month_trans = trans[(trans['Date'].dt.month == now.month) & (trans['Date'].dt.year == now.year)]
    expenses = current_month_trans[current_month_trans['Amount'] < 0].copy()
    expenses['Amount'] = expenses['Amount'].abs()

    spending_by_cat = expenses.groupby('Category')['Amount'].sum()

    budget_status = []
    for cat, limit in budgets.items():
        spent = spending_by_cat.get(cat, 0)
        remaining = limit - spent
        progress = (spent / limit) * 100 if limit > 0 else 0
        budget_status.append({
            "Category": cat, "Budget": limit, "Spent": spent,
            "Remaining": remaining, "Progress": progress
        })

    if not budget_status:
        st.info("No budgets set. Go to settings to add some.")
        return

    df_status = pd.DataFrame(budget_status)
    st.subheader("Your Monthly Budget Status")

    for index, row in df_status.iterrows():
        st.markdown(f"**{row['Category']}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Budget", f"${row['Budget']:,.2f}")
        col2.metric("Spent", f"${row['Spent']:,.2f}")
        col3.metric("Remaining", f"${row['Remaining']:,.2f}")

        progress_color = 'green' if row['Progress'] <= 75 else 'orange' if row['Progress'] <= 100 else 'red'
        st.markdown(f"""
        <style>
            .stProgress > div > div > div > div {{
                background-color: {progress_color};
            }}
        </style>""", unsafe_allow_html=True)
        st.progress(int(min(row['Progress'], 100)))
        st.markdown("---")

    with st.expander("✏️ Manage Budget Categories"):
        with st.form("budget_form"):
            new_budgets = budgets.copy()
            for cat, limit in budgets.items():
                new_limit = st.number_input(f"Budget for {cat} ($)", value=limit, min_value=0)
                new_budgets[cat] = new_limit

            if st.form_submit_button("Update Budgets"):
                st.session_state.user_data['budgets'] = new_budgets
                save_data()
                st.success("Budgets updated!")
                st.rerun()

def goals_page():
    st.title("🎯 Financial Goals")
    st.markdown("Set and track your financial goals, from saving for a vacation to a down payment.")

    goals = st.session_state.user_data.get('goals', [])

    if not goals:
        st.info("You haven't set any goals yet. Add one below!")
    else:
        for i, goal in enumerate(goals):
            with st.container(border=True):
                st.subheader(goal['name'])
                progress = (goal['current'] / goal['target']) * 100 if goal['target'] > 0 else 0
                st.progress(int(progress))
                col1, col2, col3 = st.columns(3)
                col1.metric("Target Amount", f"${goal['target']:,.2f}")
                col2.metric("Current Savings", f"${goal['current']:,.2f}")
                col3.metric("Remaining", f"${goal['target'] - goal['current']:,.2f}")
                if st.button(f"❌ Delete Goal '{goal['name']}'", key=f"del_goal_{i}"):
                    goals.pop(i)
                    save_data()
                    st.rerun()
            st.markdown("---")

    with st.expander("➕ Add a New Goal"):
        with st.form("goal_form", clear_on_submit=True):
            goal_name = st.text_input("Goal Name (e.g., Vacation Fund)")
            target_amount = st.number_input("Target Amount ($)", min_value=1.0, step=500.0)
            current_amount = st.number_input("Current Amount Saved ($)", min_value=0.0, step=100.0)
            if st.form_submit_button("Add Goal"):
                if not goal_name or target_amount <= 0:
                    st.warning("Please enter a valid goal name and target amount.")
                else:
                    goals.append({"name": goal_name, "target": target_amount, "current": current_amount})
                    save_data()
                    st.success(f"Goal '{goal_name}' added!")
                    st.rerun()

def tax_estimator_page():
    st.title("🧾 U.S. Federal Tax Estimator")
    st.warning("**Disclaimer:** This is a simplified estimator for informational purposes only. It is not financial advice. Consult a tax professional for accurate calculations.")

    # 2024 Tax Brackets (example, use current year's)
    brackets_single = {0: 0.10, 11600: 0.12, 47150: 0.22, 100525: 0.24, 191950: 0.32, 243725: 0.35, 609350: 0.37}
    deduction_single = 14600

    def calculate_federal_tax(income, brackets, deduction):
        taxable_income = max(0, income - deduction)
        tax = 0
        previous_bracket = 0
        for bracket, rate in brackets.items():
            if taxable_income > bracket:
                tax += (min(taxable_income, list(brackets.keys())[list(brackets.keys()).index(bracket)+1] if list(brackets.keys()).index(bracket)+1 < len(brackets) else taxable_income) - bracket) * rate
            else:
                break
        return tax, taxable_income

    with st.form("tax_form"):
        st.subheader("Your Financials")
        gross_income = st.number_input("Estimated Annual Gross Income ($)", min_value=0, value=75000)
        pre_tax_deductions = st.number_input("Estimated Pre-Tax Deductions (401k, HSA, etc.) ($)", min_value=0, value=5000)

        submitted = st.form_submit_button("Estimate My Taxes")

    if submitted:
        agi = gross_income - pre_tax_deductions
        estimated_tax, taxable_income = calculate_federal_tax(agi, brackets_single, deduction_single)
        effective_rate = (estimated_tax / agi * 100) if agi > 0 else 0

        st.subheader("Tax Estimation Results (Single Filer)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Adjusted Gross Income", f"${agi:,.2f}")
        col2.metric("Taxable Income", f"${taxable_income:,.2f}")
        col3.metric("Estimated Tax Liability", f"${estimated_tax:,.2f}")
        col4.metric("Effective Tax Rate", f"{effective_rate:.2f}%")

def debt_calculator_page():
    st.title("❄️ Debt Paydown Calculator")
    st.markdown("Strategize your debt repayment using the Avalanche or Snowball method.")

    debts = st.session_state.user_data.get('debts', [])

    if debts:
        st.subheader("Your Debts")
        st.dataframe(pd.DataFrame(debts), use_container_width=True)

        extra_payment = st.number_input("Extra Monthly Payment ($)", min_value=0.0, value=200.0, step=50.0)

        if st.button("Calculate Paydown Plans"):
            # Snowball: Sort by balance ascending
            snowball_debts = sorted(debts, key=lambda x: x['principal'])
            # Avalanche: Sort by interest rate descending
            avalanche_debts = sorted(debts, key=lambda x: x['rate'], reverse=True)

            def simulate(debt_list, extra):
                total_paid = 0
                months = 0
                schedule = []
                temp_debts = [d.copy() for d in debt_list]

                while any(d['principal'] > 0 for d in temp_debts):
                    months += 1
                    monthly_total_payment = sum(d['min_payment'] for d in temp_debts) + extra

                    for debt in temp_debts:
                        interest = (debt['principal'] * (debt['rate']/100)) / 12
                        debt['principal'] += interest

                    paid_this_month = 0
                    for debt in temp_debts:
                        if debt['principal'] > 0:
                            payment = min(debt['principal'], debt['min_payment'])
                            debt['principal'] -= payment
                            paid_this_month += payment

                    remaining_payment = monthly_total_payment - paid_this_month

                    # Apply extra payments based on strategy order
                    for debt in temp_debts:
                        if debt['principal'] > 0 and remaining_payment > 0:
                            payment = min(debt['principal'], remaining_payment)
                            debt['principal'] -= payment
                            remaining_payment -= payment

                    total_paid += monthly_total_payment - remaining_payment

                return months, total_paid

            snowball_months, snowball_paid = simulate(snowball_debts, extra_payment)
            avalanche_months, avalanche_paid = simulate(avalanche_debts, extra_payment)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Avalanche Method (Highest Interest First)")
                st.metric("Time to Debt Free", f"{avalanche_months} months")
                st.metric("Total Interest Paid", f"${avalanche_paid - sum(d['principal'] for d in debts):,.2f}")
            with col2:
                st.subheader("Snowball Method (Lowest Balance First)")
                st.metric("Time to Debt Free", f"{snowball_months} months")
                st.metric("Total Interest Paid", f"${snowball_paid - sum(d['principal'] for d in debts):,.2f}")

    with st.expander("➕ Add or Manage Debts"):
        with st.form("debt_form", clear_on_submit=True):
            debt_name = st.text_input("Debt Name (e.g., Credit Card)")
            principal = st.number_input("Principal ($)", min_value=0.01)
            rate = st.number_input("Interest Rate (%)", min_value=0.01, max_value=100.0)
            min_payment = st.number_input("Minimum Monthly Payment ($)", min_value=0.01)

            if st.form_submit_button("Add Debt"):
                if debt_name and principal > 0 and rate > 0 and min_payment > 0:
                    debts.append({"name": debt_name, "principal": principal, "rate": rate, "min_payment": min_payment})
                    save_data()
                    st.rerun()
                else:
                    st.warning("Please fill all fields with valid values.")

def settings_page():
    st.title("⚙️ Settings & Data Management")

    profile = st.session_state.user_data.get('profile', {})
    with st.form("profile_form"):
        st.subheader("Update Your Profile")
        name = st.text_input("Your Name", value=profile.get('name', ''))
        income = st.number_input("Net Monthly Income ($)", value=profile.get('monthly_income', 5000), min_value=0)
        if st.form_submit_button("Save Changes"):
            st.session_state.user_data['profile'] = {"name": name, "monthly_income": income}
            save_data()
            st.success("Profile updated!")

    st.markdown("---")
    st.subheader("Data Management")
    col1, col2 = st.columns(2)
    with col1:
        # Export data
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "rb") as fp:
                st.download_button(
                    label="📥 Export My Data",
                    data=fp,
                    file_name="finance_copilot_data.json",
                    mime="application/json"
                )
    with col2:
        # Import data
        uploaded_file = st.file_uploader("📤 Import Data File", type=['json'])
        if uploaded_file is not None:
            load_data(uploaded_file)
            save_data()
            st.success("Data imported successfully!")
            st.rerun()

    with st.expander("📥 Import Transactions from CSV"):
        csv_file = st.file_uploader("Upload your bank statement CSV", type=['csv'])
        if csv_file:
            df = pd.read_csv(csv_file)
            st.write("Preview of your CSV:")
            st.dataframe(df.head())

            st.write("Please map the columns from your file.")
            col1, col2, col3, col4 = st.columns(4)
            date_col = col1.selectbox("Date Column", df.columns)
            desc_col = col2.selectbox("Description Column", df.columns)
            amount_col = col3.selectbox("Amount Column", df.columns)
            cat_col = col4.selectbox("Category Column (Optional)", [None] + list(df.columns))