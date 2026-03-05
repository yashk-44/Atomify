import streamlit as st
# from backend.login import authenticate, create_account


def login_page():
    """Display login page."""
    st.title("Atomify ⚛️", text_alignment="center") # hello

    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.success(f"Welcome back, {username}!")
        else:
            st.warning("Please enter both username and password")

    st.markdown("---")
    if st.button("Don't have an account? Sign up"):
        st.session_state.page = "signup"
        st.rerun()


def signup_page():
    """Display signup page."""
    st.title("Atomify ⚛️", text_alignment="center")
    st.subheader("📝 Sign Up")

    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if not all([username, email, password, confirm_password]):
            st.warning("Please fill in all fields")
        elif password != confirm_password:
            st.error("Passwords don't match")
        else:
            st.success("Account created successfully!")
            st.info(f"Welcome, {username}!")

    st.markdown("---")
    if st.button("Already have an account? Login"):
        st.session_state.page = "login"
        st.rerun()


def authenticate(username, password):
    # TEMP: replace later with DB lookup
    return username == "demo" and password == "demo"
