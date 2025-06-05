import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the pages
def user_overview_analysis():
    st.title("📈 User Overview Analysis")
    st.write("This page will contain user overview analytics.")
    
    # Placeholder content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", "1,234", "12%")
    with col2:
        st.metric("Active Users", "987", "8%")
    with col3:
        st.metric("New Users", "156", "23%")
    
    st.info("🚧 Page content will be developed here")

def user_engagement_analysis():
    st.title("🎯 User Engagement Analysis")
    st.write("This page will contain user engagement analytics.")
    
    # Placeholder content
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Session Duration", "12.5 min", "5%")
    with col2:
        st.metric("Page Views per Session", "4.2", "2%")
    
    st.info("🚧 Page content will be developed here")

def experience_analysis():
    st.title("🌟 Experience Analysis")
    st.write("This page will contain user experience analytics.")
    
    # Placeholder content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Bounce Rate", "32%", "-3%")
    with col2:
        st.metric("Load Time", "2.1s", "-0.3s")
    with col3:
        st.metric("Error Rate", "0.8%", "-0.2%")
    
    st.info("🚧 Page content will be developed here")

def satisfaction_analysis():
    st.title("😊 Satisfaction Analysis")
    st.write("This page will contain user satisfaction analytics.")
    
    # Placeholder content
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Customer Satisfaction", "4.2/5", "0.1")
    with col2:
        st.metric("NPS Score", "67", "5")
    
    st.info("🚧 Page content will be developed here")

# Sidebar navigation
def main():
    st.sidebar.title("📊 Analytics Dashboard")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Navigate to:",
        [
            "User Overview Analysis",
            "User Engagement Analysis", 
            "Experience Analysis",
            "Satisfaction Analysis"
        ],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("This dashboard provides comprehensive analytics across user behavior, engagement, experience, and satisfaction metrics.")
    
    # Route to the selected page
    if page == "User Overview Analysis":
        user_overview_analysis()
    elif page == "User Engagement Analysis":
        user_engagement_analysis()
    elif page == "Experience Analysis":
        experience_analysis()
    elif page == "Satisfaction Analysis":
        satisfaction_analysis()

if __name__ == "__main__":
    main()