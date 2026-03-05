import streamlit as st
from pages.login import *
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

st.set_page_config(
    page_title="Atomify",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="collapsedControl"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = "login"

    # Route to appropriate page
if st.session_state.page == "login":
    login_page()
else:
    signup_page()
