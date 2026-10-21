import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Smart Rent Prediction",
    page_icon="🏠",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = joblib.load("rent_model.pkl")
    features = joblib.load("rent_model_features.pkl")
    return model, features


# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    data = pd.read_csv("cities_magicbricks_rental_prices.csv")

    for column in ["city", "furnishing", "house_type", "locality"]:
        if column in data.columns:
            data[column] = data[column].astype("string").str.strip()

    for column in ["beds", "bathrooms", "balconies", "area", "rent"]:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    return data


# -------------------- INITIALIZE --------------------
try:
    model, model_features = load_model()
    df = load_data()
except Exception as error:
    st.error(f"Project files load nahi ho paaye: {error}")
    st.stop()


# -------------------- UI STYLING --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #0f0f0f, #1c1c1c);
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid #333333;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }

    .stButton > button {
        background: linear-gradient(45deg, #ff4b2b, #ff416c);
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
    }

    [data-testid="stMetric"] {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------- TITLE --------------------
st.title("🏠 Smart Rent Prediction Dashboard")
st.caption(
    "Estimate monthly house rent using a trained machine learning model."
)


# -------------------- VALIDATE DATA --------------------
required_columns = [
    "city", "furnishing", "beds", "bathrooms",
    "balconies", "area", "rent"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Dataset mein columns missing hain: {missing_columns}")
    st.stop()

if df.empty:
    st.error("Dataset empty hai. CSV file check karo.")
    st.stop()


# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("🔍 Property Details")

bhk = st.sidebar.slider(
    "BHK",
    min_value=1,
    max_value=5,
    value=2
)

area = st.sidebar.number_input(
    "Area (sqft)",
    min_value=100,
    max_value=10000,
    value=500,
    step=50
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    min_value=1,
    max_value=5,
    value=2
)

balconies = st.sidebar.slider(
    "Balconies",
    min_value=0,
    max_value=5,
    value=1
)

cities = sorted(
    df["city"].dropna().unique().tolist()
)

if not cities:
    st.error("Dataset mein city values nahi hain.")
    st.stop()

city = st.sidebar.selectbox("City", cities)

furnishing_options = sorted(
    df["furnishing"].dropna().unique().tolist()
)

if not furnishing_options:
    st.error("Dataset mein furnishing values nahi hain.")
    st.stop()

furnishing = st.sidebar.selectbox(
    "Furnishing",
    furnishing_options
)


# -------------------- HOUSE / LISTING SELECTION --------------------
if "house_type" in df.columns:
    city_rows = df[df["city"] == city].copy()

    city_bhk_rows = city_rows[
        city_rows["beds"] == bhk
    ]

    if not city_bhk_rows.empty:
        listing_rows = city_bhk_rows
    else:
        listing_rows = city_rows

    house_types = sorted(
        listing_rows["house_type"].dropna().unique().tolist()
    )

    if house_types:
        house_type = st.sidebar.selectbox(
            "House / Listing",
            house_types
        )

        selected_rows = listing_rows[
            listing_rows["house_type"] == house_type
        ]

        if not selected_rows.empty:
            selected_locality = str(
                selected_rows.iloc[0].get("locality", "")
            )
        else:
            selected_locality = ""

        st.sidebar.caption(
            f"📍 Selected locality: {selected_locality}"
        )
    else:
        house_type = None
        selected_locality = ""
        st.sidebar.warning(
            "Is city mein house/listing information available nahi hai."
        )
else:
    house_type = None
    selected_locality = ""


# -------------------- PROPERTY SUMMARY --------------------
st.subheader("📊 Property Summary")

col1, col2, col3 = st.columns(3)

col1.metric("🏠 BHK", bhk)
col2.metric("📐 Area", f"{area:,} sqft")
col3.metric("🚿 Bathrooms", bathrooms)

col4, col5, col6 = st.columns(3)

col4.metric("📍 City", city)
col5.metric("🛋 Furnishing", furnishing)
col6.metric("🌿 Balconies", balconies)


# -------------------- PREDICTION --------------------
if st.button("💰 Predict Rent", type="primary"):

    try:
        # Model ke expected features aur order mein input banao
        input_data = pd.DataFrame(
            np.zeros((1, len(model_features))),
            columns=model_features
        )

        # Numeric input values
        numeric_values = {
            "area": area,
            "beds": bhk,
            "bathrooms": bathrooms,
            "balconies": balconies
        }

        for feature, value in numeric_values.items():
            if feature in input_data.columns:
                input_data.loc[0, feature] = value

        # Categorical features
        category_values = {
            "city": city,
            "furnishing": furnishing,
            "locality": selected_locality
        }

        if house_type is not None:
            category_values["house_type"] = house_type

        for prefix, value in category_values.items():
            feature_name = f"{prefix}_{value}"

            if feature_name in input_data.columns:
                input_data.loc[0, feature_name] = 1

        # House-type feature names ke liye alternate mapping
        # Agar model mein property_type_* columns hain
        if house_type is not None:
            property_type_feature = f"property_type_{house_type}"

            if property_type_feature in input_data.columns:
                input_data.loc[0, property_type_feature] = 1

        # Exact model feature order
        input_data = input_data[model_features]

        # Predict
        predicted_rent = float(
            model.predict(input_data)[0]
        )

        predicted_rent = max(0, predicted_rent)

        st.success(
            f"💰 Estimated Monthly Rent: ₹ {predicted_rent:,.0f}"
        )

        st.caption(
            "Ye machine learning model ka estimate hai, "
            "guaranteed market price nahi."
        )

        # -------------------- SIMILAR LISTINGS --------------------
        st.subheader(f"🏘 Similar Listings in {city}")

        # Exact city, BHK aur bathroom match
        rec = df[
            (df["city"] == city)
            & (df["beds"] == bhk)
            & (df["bathrooms"] == bathrooms)
            & df["rent"].notna()
        ].copy()

        # Fallback: city aur BHK
        if rec.empty:
            rec = df[
                (df["city"] == city)
                & (df["beds"] == bhk)
                & df["rent"].notna()
            ].copy()

        # Fallback: selected city
        if rec.empty:
            rec = df[
                (df["city"] == city)
                & df["rent"].notna()
            ].copy()

        if not rec.empty:
            # Missing area values ko handle karo
            rec = rec.dropna(subset=["area", "rent"])

        if not rec.empty:
            rec["area_difference"] = (
                rec["area"] - area
            ).abs()

            rec["rent_difference"] = (
                rec["rent"] - predicted_rent
            ).abs()

            rec["furnishing_match"] = (
                rec["furnishing"] == furnishing
            ).astype(int)

            # Matching furnishing, area aur rent ke basis par ranking
            rec = rec.sort_values(
                by=[
                    "furnishing_match",
                    "area_difference",
                    "rent_difference"
                ],
                ascending=[False, True, True]
            ).head(6)

            # Similar property cards
            for _, row in rec.iterrows():
                locality_value = row.get("locality", "Unknown")
                furnishing_value = row.get("furnishing", "Unknown")

                beds_value = row.get("beds", 0)
                bathrooms_value = row.get("bathrooms", 0)
                area_value = row.get("area", 0)
                rent_value = row.get("rent", 0)

                st.markdown(
                    f"""
                    <div style="
                        background:#1e1e1e;
                        padding:15px;
                        border-radius:12px;
                        margin-bottom:10px;
                        border:1px solid #333;
                    ">
                        <h4>📍 {locality_value}</h4>
                        <p>{row['city']} | {furnishing_value}</p>
                        <p>
                            🏠 {beds_value:g} BHK |
                            🚿 {bathrooms_value:g} Bathrooms
                        </p>
                        <p>📐 {area_value:,.0f} sqft</p>
                        <h3>💰 ₹ {rent_value:,.0f} / month</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # -------------------- COMPARISON CHART --------------------
            st.subheader(
                "📈 Predicted Rent vs Similar Listings"
            )

            chart_data = pd.DataFrame({
                "Property": ["Model Estimate"] + [
                    f"Listing {i + 1}"
                    for i in range(len(rec))
                ],
                "Monthly Rent (₹)": [
                    predicted_rent
                ] + rec["rent"].tolist()
            })

            st.bar_chart(
                chart_data.set_index("Property")
            )

            st.caption(
                "Model estimate aur dataset mein recorded rents ka comparison. "
                "Listing rents historical dataset values hain, live market prices nahi."
            )

        else:
            st.info(
                "Selected city ke liye property listings nahi mili."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

        st.info(
            "Model, feature file aur dataset compatible hain ya nahi, "
            "check karo. Model ki feature names bhi verify karni pad sakti hain."
        )
