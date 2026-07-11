
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("🌾 Agricultural Data Explorer")
st.markdown("--- ")

st.write("### Explore your agricultural datasets here.")

# Load the dummy data (or allow user to upload)
# In a real application, you might use st.file_uploader here
try:
    df = pd.read_csv('dummy_crop_data.csv')
    st.success("Dummy crop data loaded successfully!")

    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Data Description")
    st.write(df.describe())

    st.subheader("Crop Yields by Region")
    # Simple aggregation and display
    yield_by_region = df.groupby('Region')['Yield_Tons_Per_Acre'].mean().sort_values(ascending=False)
    st.bar_chart(yield_by_region)

except FileNotFoundError:
    st.error("Error: 'dummy_crop_data.csv' not found. Please ensure it exists in the main directory.")
except Exception as e:
    st.error(f"An error occurred while loading or processing data: {e}")


st.markdown("--- ")
st.info("This is a sample data page. You can customize it to load and visualize your specific agricultural datasets.")
