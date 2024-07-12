import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load the dataset
data = pd.read_csv('./data/MSCI 436 - Project Dataset.csv')

# Create "Age of Company (Years)" feature
current_year = pd.Timestamp.now().year
data["Age of Company (Years)"] = current_year - data["Founding Year"]

# Keep a copy of the original data with the Company Name
original_data = data.copy()

# Drop unnecessary columns for model training
data.drop(columns=["Company Name", "Founding Year", "Sector", "Industry", "Country"], inplace=True)

# Ordinal encoding for "Risk Level"
risk_level_map = {"Low": 1, "Medium": 2, "High": 3}
data["Risk Level"] = data["Risk Level"].map(risk_level_map)

# Select only numerical features for training
numerical_features = ["Risk Level", "Annual Revenue 2022-2023 (USD in Billions)", "Market Cap (USD in Billions)",
                      "Employee Size", "Dividend Yield", "% Growth over last year", "Age of Company (Years)"]

# Scale numerical features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[numerical_features])

# Fit the KNN model with n_neighbors=5
knn = NearestNeighbors(n_neighbors=5)
knn.fit(data_scaled)

# Function to recommend stocks based on user input and show original feature values
def recommend_stocks(user_input, n_recommendations=5):
    # Process user input to match the data format
    user_input["Risk Level"] = user_input["Risk Level"].map(risk_level_map)
    user_input_scaled = scaler.transform(user_input[numerical_features])
    
    # Find the nearest neighbors
    distances, indices = knn.kneighbors(user_input_scaled, n_neighbors=n_recommendations)
    
    # Retrieve the original data of the recommended companies
    recommended_companies = original_data.iloc[indices[0]]
    return recommended_companies

# Streamlit UI
st.set_page_config(page_title='StockPickr', page_icon='💵', layout='wide', initial_sidebar_state='expanded')

# Custom CSS for the theme
st.markdown("""
    <style>
        .stApp {
            background-color: #d0e8d0;
            color: #000000;
        }
        .title {
            font-size: 3em;
            color: #2e7d32;
            text-align: center;
        }
        .subtitle {
            font-size: 1.5em;
            color: #2e7d32;
            text-align: center;
        }
        .blurb {
            font-size: 1.2em;
            color: #000000;
            text-align: center;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 1.8em;
            color: #2e7d32;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .stock-recommendation {
            background-color: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .stock-recommendation .feature {
            font-size: 1.1em;
            color: #1b5e20;
        }
        .sidebar .sidebar-content {
            background-color: #e8f5e9;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">StockPickr</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">We get it, investment decisions are not easy. Let us help you with your decision making, and recommend some stocks to invest into!</p>', unsafe_allow_html=True)

# User input
st.sidebar.header('User Input Features')
st.sidebar.markdown('<p class="blurb">Please enter your preferences for the stock you would like to invest into. Move the sliders for each feature to enter your preferred description of a company you would like to invest into. Let us do the rest!</p>', unsafe_allow_html=True)

def user_input_features():
    industry = st.sidebar.selectbox('Industry', original_data['Industry'].unique())
    country = st.sidebar.selectbox('Country', original_data['Country'].unique())
    risk_level = st.sidebar.selectbox('Risk Level', ['Low', 'Medium', 'High'])
    
    age_of_company = st.sidebar.slider('Age of Company (Years)', min_value=0, max_value=100, value=20)
    
    revenue_min = float(original_data['Annual Revenue 2022-2023 (USD in Billions)'].quantile(0.05))
    revenue_max = float(original_data['Annual Revenue 2022-2023 (USD in Billions)'].quantile(0.95))
    annual_revenue = st.sidebar.slider('Annual Revenue 2022-2023 (USD in Billions)', min_value=revenue_min, max_value=revenue_max, value=50.0, format="$%.2f")
    
    market_cap_min = float(original_data['Market Cap (USD in Billions)'].quantile(0.05))
    market_cap_max = float(original_data['Market Cap (USD in Billions)'].quantile(0.95))
    market_cap = st.sidebar.slider('Market Cap (USD in Billions)', min_value=market_cap_min, max_value=market_cap_max, value=200.0, format="$%.2f")
    
    employee_size_min = int(original_data['Employee Size'].quantile(0.05))
    employee_size_max = int(original_data['Employee Size'].quantile(0.95))
    employee_size = st.sidebar.slider('Employee Size (employees)', min_value=employee_size_min, max_value=employee_size_max, value=10000, format="%d employees")
    
    dividend_yield = st.sidebar.slider('Dividend Yield (%)', min_value=0.0, max_value=8.0, value=2.0)
    
    growth_min = float(original_data['% Growth over last year'].quantile(0.05)) * 100
    growth_max = float(original_data['% Growth over last year'].quantile(0.95)) * 100
    growth = st.sidebar.slider('% Growth over last year', min_value=growth_min, max_value=growth_max, value=120.0)
    
    data_input = {
        'Risk Level': risk_level,
        'Annual Revenue 2022-2023 (USD in Billions)': annual_revenue,
        'Market Cap (USD in Billions)': market_cap,
        'Employee Size': employee_size,
        'Dividend Yield': dividend_yield / 100,
        '% Growth over last year': growth / 100,
        'Age of Company (Years)': age_of_company
    }
    return pd.DataFrame([data_input])

user_input = user_input_features()

# Display user input
st.markdown('<h2 class="section-title">User Input</h2>', unsafe_allow_html=True)
st.write(user_input)

# Get recommendations
recommended_companies = recommend_stocks(user_input, n_recommendations=5)

# Display recommendations
st.markdown('<h2 class="section-title">Recommended Stocks and Their Feature Values</h2>', unsafe_allow_html=True)
st.markdown('<p class="blurb">Based on your responses, we recommend you to invest in these 5 stocks! Here is their respective descriptions:</p>', unsafe_allow_html=True)
for index, row in recommended_companies.iterrows():
    st.markdown(f'<div class="stock-recommendation">', unsafe_allow_html=True)
    st.markdown(f'<strong>Company:</strong> {row["Company Name"]}', unsafe_allow_html=True)
    st.markdown(f'<div class="feature"><strong>Industry:</strong> {row["Industry"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="feature"><strong>Sector:</strong> {row["Sector"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="feature"><strong>Country:</strong> {row["Country"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="feature"><strong>Risk Level:</strong> {row["Risk Level"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="feature"><strong>Age of Company (Years):</strong> {row["Age of Company (Years)"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="feature"><strong>Annual Revenue 2022-2023 (USD in Billions):</strong> ${row["Annual Revenue 2022-2023 (USD in Billions)"]:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="feature"><strong>Market Cap (USD in Billions):</strong> ${row["Market Cap (USD in Billions)"]:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="feature"><strong>Employee Size:</strong> {row["Employee Size"]} employees</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="feature"><strong>Dividend Yield:</strong> {row["Dividend Yield"]:.2%}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="feature"><strong>% Growth over last year:</strong> {row["% Growth over last year"]:.2%}</div>', unsafe_allow_html=True)