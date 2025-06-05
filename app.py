import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure the page
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process the telecom data"""
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Convert column names to lowercase with underscores
        df.columns = [column.replace(' ', '_').lower() for column in df.columns]
        
        # Basic data cleaning
        if 'start' in df.columns:
            df['start'] = pd.to_datetime(df['start'], errors='coerce')
        if 'end' in df.columns:
            df['end'] = pd.to_datetime(df['end'], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Define the pages
def user_overview_analysis():
    st.title("📈 User Overview Analysis")
    st.markdown("---")
    
    # Check if data is uploaded
    uploaded_file = st.file_uploader("Upload your telecom dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        # Load and process data
        df = load_and_process_data(uploaded_file)
        
        if df is not None:
            # Display dataset overview
            st.subheader("📊 Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columns", f"{df.shape[1]:,}")
            with col3:
                unique_users = df['msisdn/number'].nunique() if 'msisdn/number' in df.columns else 0
                st.metric("Unique Users", f"{unique_users:,}")
            with col4:
                missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing Data %", f"{missing_percent:.2f}%")
            
            st.markdown("---")
            
            # Data Quality Analysis
            st.subheader("🔍 Data Quality Analysis")
            
            # Missing values analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Missing Values by Column**")
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                
                if len(missing_data) > 0:
                    missing_df = pd.DataFrame({
                        'Column': missing_data.index,
                        'Missing Count': missing_data.values,
                        'Missing %': (missing_data.values / len(df)) * 100
                    })
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("No missing values found!")
            
            with col2:
                if len(missing_data) > 0:
                    fig_missing = px.bar(
                        x=missing_data.values[:10], 
                        y=missing_data.index[:10],
                        orientation='h',
                        title="Top 10 Columns with Missing Values",
                        labels={'x': 'Missing Count', 'y': 'Columns'}
                    )
                    fig_missing.update_layout(height=400)
                    st.plotly_chart(fig_missing, use_container_width=True)
            
            st.markdown("---")
            
            # Handset Analysis
            st.subheader("📱 Handset Analysis")
            
            if 'handset_manufacturer' in df.columns and 'handset_type' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Top Handset Manufacturers**")
                    top_manufacturers = df['handset_manufacturer'].value_counts().head(10)
                    
                    fig_manufacturers = px.bar(
                        x=top_manufacturers.values,
                        y=top_manufacturers.index,
                        orientation='h',
                        title="Top 10 Handset Manufacturers",
                        labels={'x': 'Count', 'y': 'Manufacturer'}
                    )
                    fig_manufacturers.update_layout(height=400)
                    st.plotly_chart(fig_manufacturers, use_container_width=True)
                
                with col2:
                    st.write("**Top Handset Types**")
                    top_handsets = df['handset_type'].value_counts().head(10)
                    
                    fig_handsets = px.bar(
                        x=top_handsets.values,
                        y=top_handsets.index,
                        orientation='h',
                        title="Top 10 Handset Types",
                        labels={'x': 'Count', 'y': 'Handset Type'}
                    )
                    fig_handsets.update_layout(height=400)
                    st.plotly_chart(fig_handsets, use_container_width=True)
                
                # Top handsets by manufacturer
                st.write("**Top Handsets by Manufacturer**")
                manufacturer_options = ['Apple', 'Samsung', 'Huawei'] + list(top_manufacturers.head(3).index)
                manufacturer_options = list(set(manufacturer_options))  # Remove duplicates
                
                selected_manufacturer = st.selectbox("Select Manufacturer", manufacturer_options)
                
                if selected_manufacturer in df['handset_manufacturer'].values:
                    manufacturer_handsets = df[df['handset_manufacturer'] == selected_manufacturer]['handset_type'].value_counts().head(10)
                    
                    if len(manufacturer_handsets) > 0:
                        fig_manu_handsets = px.bar(
                            x=manufacturer_handsets.values,
                            y=manufacturer_handsets.index,
                            orientation='h',
                            title=f"Top Handsets by {selected_manufacturer}",
                            labels={'x': 'Count', 'y': 'Handset Type'}
                        )
                        st.plotly_chart(fig_manu_handsets, use_container_width=True)
                    else:
                        st.warning(f"No data found for {selected_manufacturer}")
            
            st.markdown("---")
            
            # User Behavior Analysis
            st.subheader("👥 User Behavior Analysis")
            
            if 'msisdn/number' in df.columns:
                # Session analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'bearer_id' in df.columns:
                        sessions = df.groupby('msisdn/number')['bearer_id'].count().reset_index()
                        sessions.columns = ['User', 'Sessions']
                        sessions = sessions.sort_values('Sessions', ascending=False)
                        
                        st.write("**Session Statistics**")
                        session_stats = sessions['Sessions'].describe()
                        st.dataframe(session_stats.to_frame().T, use_container_width=True)
                        
                        # Top users by sessions
                        st.write("**Top 10 Users by Sessions**")
                        st.dataframe(sessions.head(10), use_container_width=True)
                
                with col2:
                    if 'dur._(ms)' in df.columns:
                        durations = df.groupby('msisdn/number')['dur._(ms)'].sum().reset_index()
                        durations.columns = ['User', 'Total_Duration_ms']
                        durations = durations.sort_values('Total_Duration_ms', ascending=False)
                        
                        st.write("**Duration Statistics**")
                        duration_stats = durations['Total_Duration_ms'].describe()
                        st.dataframe(duration_stats.to_frame().T, use_container_width=True)
                        
                        # Top users by duration
                        st.write("**Top 10 Users by Duration**")
                        durations_display = durations.copy()
                        durations_display['Duration_Hours'] = durations_display['Total_Duration_ms'] / (1000 * 60 * 60)
                        st.dataframe(durations_display[['User', 'Duration_Hours']].head(10), use_container_width=True)
            
            st.markdown("---")
            
            # Data Usage Analysis
            st.subheader("📊 Data Usage Analysis")
            
            # Calculate total data usage
            data_columns = ['social_media_dl_(bytes)', 'social_media_ul_(bytes)', 'google_dl_(bytes)', 
                          'google_ul_(bytes)', 'email_dl_(bytes)', 'email_ul_(bytes)', 'youtube_dl_(bytes)', 
                          'youtube_ul_(bytes)', 'netflix_dl_(bytes)', 'netflix_ul_(bytes)', 'gaming_dl_(bytes)', 
                          'gaming_ul_(bytes)', 'other_dl_(bytes)', 'other_ul_(bytes)']
            
            available_data_cols = [col for col in data_columns if col in df.columns]
            
            if available_data_cols:
                # Create application totals
                app_data = {}
                if 'social_media_dl_(bytes)' in df.columns and 'social_media_ul_(bytes)' in df.columns:
                    app_data['Social Media'] = (df['social_media_dl_(bytes)'] + df['social_media_ul_(bytes)']).sum()
                if 'google_dl_(bytes)' in df.columns and 'google_ul_(bytes)' in df.columns:
                    app_data['Google'] = (df['google_dl_(bytes)'] + df['google_ul_(bytes)']).sum()
                if 'email_dl_(bytes)' in df.columns and 'email_ul_(bytes)' in df.columns:
                    app_data['Email'] = (df['email_dl_(bytes)'] + df['email_ul_(bytes)']).sum()
                if 'youtube_dl_(bytes)' in df.columns and 'youtube_ul_(bytes)' in df.columns:
                    app_data['YouTube'] = (df['youtube_dl_(bytes)'] + df['youtube_ul_(bytes)']).sum()
                if 'netflix_dl_(bytes)' in df.columns and 'netflix_ul_(bytes)' in df.columns:
                    app_data['Netflix'] = (df['netflix_dl_(bytes)'] + df['netflix_ul_(bytes)']).sum()
                if 'gaming_dl_(bytes)' in df.columns and 'gaming_ul_(bytes)' in df.columns:
                    app_data['Gaming'] = (df['gaming_dl_(bytes)'] + df['gaming_ul_(bytes)']).sum()
                if 'other_dl_(bytes)' in df.columns and 'other_ul_(bytes)' in df.columns:
                    app_data['Other'] = (df['other_dl_(bytes)'] + df['other_ul_(bytes)']).sum()
                
                if app_data:
                    # Convert bytes to GB for better readability
                    app_data_gb = {k: v / (1024**3) for k, v in app_data.items()}
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart
                        app_df = pd.DataFrame(list(app_data_gb.items()), columns=['Application', 'Data_GB'])
                        app_df = app_df.sort_values('Data_GB', ascending=False)
                        
                        fig_apps = px.bar(
                            app_df, 
                            x='Data_GB', 
                            y='Application',
                            orientation='h',
                            title="Total Data Usage by Application (GB)",
                            labels={'Data_GB': 'Data Usage (GB)', 'Application': 'Application'}
                        )
                        st.plotly_chart(fig_apps, use_container_width=True)
                    
                    with col2:
                        # Pie chart
                        fig_pie = px.pie(
                            app_df, 
                            values='Data_GB', 
                            names='Application',
                            title="Data Usage Distribution by Application"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
            
            # Data quality summary
            st.markdown("---")
            st.subheader("✅ Data Processing Summary")
            
            processing_steps = [
                "✅ Data loaded successfully",
                "✅ Column names standardized",
                "✅ Missing values analyzed",
                "✅ Handset analysis completed",
                "✅ User behavior metrics calculated",
                "✅ Data usage patterns identified"
            ]
            
            for step in processing_steps:
                st.write(step)
            
            # Raw data preview
            if st.checkbox("Show Raw Data Preview"):
                st.subheader("📋 Raw Data Preview")
                st.dataframe(df.head(100), use_container_width=True)
                
    else:
        st.info("👆 Please upload a CSV file to begin the analysis")
        
        # Show expected data format
        st.subheader("📋 Expected Data Format")
        st.write("Your CSV file should contain columns such as:")
        expected_cols = [
            "msisdn/number", "handset_manufacturer", "handset_type", "bearer_id",
            "dur._(ms)", "social_media_dl_(bytes)", "social_media_ul_(bytes)",
            "google_dl_(bytes)", "google_ul_(bytes)", "email_dl_(bytes)", "email_ul_(bytes)",
            "youtube_dl_(bytes)", "youtube_ul_(bytes)", "netflix_dl_(bytes)", "netflix_ul_(bytes)",
            "gaming_dl_(bytes)", "gaming_ul_(bytes)", "other_dl_(bytes)", "other_ul_(bytes)"
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            for i, col in enumerate(expected_cols[:len(expected_cols)//2]):
                st.write(f"• {col}")
        with col2:
            for col in expected_cols[len(expected_cols)//2:]:
                st.write(f"• {col}")

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