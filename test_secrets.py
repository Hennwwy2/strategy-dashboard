import streamlit as st

st.title("Secrets Test Page")

st.header("Attempting to read secrets...")

# We will try to access the specific secret key
try:
    # This is the line that accesses the secret
    tiingo_key = st.secrets["tiingo"]["api_key"]

    # If the line above works, we show a success message
    st.success("✅ SUCCESS! Found the Tiingo API key in the secrets.")
    st.write("This means your secrets are configured correctly!")

except Exception as e:
    # If the line above fails, we show an error message
    st.error("❌ FAILURE: Could not read the Tiingo API key from secrets.")
    st.error("This confirms the problem is with how the secrets were entered on the Streamlit website.")
    st.info("Please go to 'Manage app' -> 'Secrets' and ensure the text is formatted correctly and saved.")
    st.code(f"The error Python gave was: {e}")