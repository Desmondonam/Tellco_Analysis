import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.cluster import KMeans
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
    st.title("📊 User Engagement Analysis")
    st.markdown("---")
    
    # Check if data is uploaded
    uploaded_file = st.file_uploader("Upload your telecom dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        # Load and process data
        df = load_and_process_data(uploaded_file)
        
        if df is not None:
            # Rename columns for consistency
            df_renamed = df.rename(columns={'msisdn/number': 'msisdn', 'dur._(ms)': 'duration'})
            
            # Calculate engagement metrics
            engagement_metrics = df_renamed.groupby('msisdn').agg({
                'bearer_id': 'count',
                'duration': 'sum', 
                'total_data': 'sum' if 'total_data' in df_renamed.columns else lambda x: 0
            })
            engagement_metrics = engagement_metrics.rename(columns={
                'bearer_id': 'sessions_frequency', 
                'total_data': 'total_traffic'
            })
            
            # Display engagement overview
            st.subheader("📈 Engagement Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Users", f"{len(engagement_metrics):,}")
            with col2:
                avg_sessions = engagement_metrics['sessions_frequency'].mean()
                st.metric("Avg Sessions/User", f"{avg_sessions:.2f}")
            with col3:
                avg_duration = engagement_metrics['duration'].mean() / (1000 * 60)  # Convert to minutes
                st.metric("Avg Duration (min)", f"{avg_duration:.2f}")
            with col4:
                if engagement_metrics['total_traffic'].sum() > 0:
                    avg_traffic = engagement_metrics['total_traffic'].mean() / (1024**2)  # Convert to MB
                    st.metric("Avg Traffic (MB)", f"{avg_traffic:.2f}")
                else:
                    st.metric("Avg Traffic", "N/A")
            
            st.markdown("---")
            
            # Top Performers Analysis
            st.subheader("🏆 Top Performers")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Top 10 by Session Frequency**")
                top_sessions = engagement_metrics.sort_values('sessions_frequency', ascending=False).head(10)
                
                fig_sessions = px.bar(
                    x=top_sessions['sessions_frequency'],
                    y=[f"User {i+1}" for i in range(len(top_sessions))],
                    orientation='h',
                    title="Top Users by Sessions",
                    labels={'x': 'Session Count', 'y': 'Users'}
                )
                fig_sessions.update_layout(height=300)
                st.plotly_chart(fig_sessions, use_container_width=True)
                
                st.dataframe(top_sessions[['sessions_frequency']], use_container_width=True)
            
            with col2:
                st.write("**Top 10 by Duration**")
                top_duration = engagement_metrics.sort_values('duration', ascending=False).head(10)
                top_duration_display = top_duration.copy()
                top_duration_display['duration_hours'] = top_duration_display['duration'] / (1000 * 60 * 60)
                
                fig_duration = px.bar(
                    x=top_duration_display['duration_hours'],
                    y=[f"User {i+1}" for i in range(len(top_duration_display))],
                    orientation='h',
                    title="Top Users by Duration (Hours)",
                    labels={'x': 'Duration (Hours)', 'y': 'Users'}
                )
                fig_duration.update_layout(height=300)
                st.plotly_chart(fig_duration, use_container_width=True)
                
                st.dataframe(top_duration_display[['duration_hours']], use_container_width=True)
            
            with col3:
                st.write("**Top 10 by Traffic**")
                if engagement_metrics['total_traffic'].sum() > 0:
                    top_traffic = engagement_metrics.sort_values('total_traffic', ascending=False).head(10)
                    top_traffic_display = top_traffic.copy()
                    top_traffic_display['traffic_gb'] = top_traffic_display['total_traffic'] / (1024**3)
                    
                    fig_traffic = px.bar(
                        x=top_traffic_display['traffic_gb'],
                        y=[f"User {i+1}" for i in range(len(top_traffic_display))],
                        orientation='h',
                        title="Top Users by Traffic (GB)",
                        labels={'x': 'Traffic (GB)', 'y': 'Users'}
                    )
                    fig_traffic.update_layout(height=300)
                    st.plotly_chart(fig_traffic, use_container_width=True)
                    
                    st.dataframe(top_traffic_display[['traffic_gb']], use_container_width=True)
                else:
                    st.info("Traffic data not available")
            
            st.markdown("---")
            
            # Statistical Summary
            st.subheader("📊 Statistical Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Engagement Metrics Statistics**")
                stats_df = engagement_metrics.describe()
                st.dataframe(stats_df, use_container_width=True)
            
            with col2:
                st.write("**Distribution Analysis**")
                
                # Distribution plots
                metric_option = st.selectbox("Select metric for distribution:", 
                                           ['sessions_frequency', 'duration', 'total_traffic'])
                
                if metric_option == 'total_traffic' and engagement_metrics['total_traffic'].sum() == 0:
                    st.warning("Traffic data not available for distribution analysis")
                else:
                    fig_dist = px.histogram(
                        engagement_metrics, 
                        x=metric_option,
                        nbins=30,
                        title=f"Distribution of {metric_option.replace('_', ' ').title()}",
                        labels={'x': metric_option.replace('_', ' ').title(), 'y': 'Frequency'}
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            st.markdown("---")
            
            # Outlier Analysis
            st.subheader("🎯 Outlier Detection")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Outlier Detection - Box Plots**")
                
                # Create box plots for each metric
                metrics_for_box = ['sessions_frequency', 'duration']
                if engagement_metrics['total_traffic'].sum() > 0:
                    metrics_for_box.append('total_traffic')
                
                selected_metric = st.selectbox("Select metric for outlier analysis:", metrics_for_box)
                
                fig_box = px.box(
                    engagement_metrics, 
                    y=selected_metric,
                    title=f"Outlier Detection - {selected_metric.replace('_', ' ').title()}",
                    labels={'y': selected_metric.replace('_', ' ').title()}
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                st.write("**Outlier Statistics**")
                
                # Calculate outlier statistics using IQR method
                def calculate_outliers(series):
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    return len(outliers), lower_bound, upper_bound
                
                outlier_stats = {}
                for metric in ['sessions_frequency', 'duration', 'total_traffic']:
                    if metric == 'total_traffic' and engagement_metrics['total_traffic'].sum() == 0:
                        continue
                    count, lower, upper = calculate_outliers(engagement_metrics[metric])
                    outlier_stats[metric] = {
                        'Outlier Count': count,
                        'Lower Bound': f"{lower:.2f}",
                        'Upper Bound': f"{upper:.2f}",
                        'Outlier %': f"{(count/len(engagement_metrics)*100):.2f}%"
                    }
                
                outlier_df = pd.DataFrame(outlier_stats).T
                st.dataframe(outlier_df, use_container_width=True)
                
                # Option to clean outliers
                if st.button("Clean Outliers"):
                    cleaned_metrics = engagement_metrics.copy()
                    
                    for col in ['sessions_frequency', 'duration', 'total_traffic']:
                        if col == 'total_traffic' and engagement_metrics['total_traffic'].sum() == 0:
                            continue
                        
                        Q1 = cleaned_metrics[col].quantile(0.25)
                        Q3 = cleaned_metrics[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        cleaned_metrics[col] = np.where(cleaned_metrics[col] < lower_bound, lower_bound, cleaned_metrics[col])
                        cleaned_metrics[col] = np.where(cleaned_metrics[col] > upper_bound, upper_bound, cleaned_metrics[col])
                    
                    st.success("Outliers cleaned successfully!")
                    st.write("**Cleaned Data Statistics:**")
                    st.dataframe(cleaned_metrics.describe(), use_container_width=True)
            
            st.markdown("---")
            
            # Clustering Analysis
            st.subheader("🎯 User Engagement Clustering")
            
            # Normalize data for clustering
            from sklearn.preprocessing import MinMaxScaler, Normalizer
            from sklearn.cluster import KMeans
            
            # Prepare data for clustering
            clustering_data = engagement_metrics.copy()
            
            # Handle outliers first
            for col in ['sessions_frequency', 'duration', 'total_traffic']:
                if col == 'total_traffic' and engagement_metrics['total_traffic'].sum() == 0:
                    continue
                
                Q1 = clustering_data[col].quantile(0.25)
                Q3 = clustering_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                clustering_data[col] = np.where(clustering_data[col] < lower_bound, lower_bound, clustering_data[col])
                clustering_data[col] = np.where(clustering_data[col] > upper_bound, upper_bound, clustering_data[col])
            
            # Normalize data
            columns_to_normalize = ['sessions_frequency', 'duration']
            if engagement_metrics['total_traffic'].sum() > 0:
                columns_to_normalize.append('total_traffic')
            
            scaler = MinMaxScaler()
            normalizer = Normalizer()
            
            scaled_data = scaler.fit_transform(clustering_data[columns_to_normalize])
            normalized_data = normalizer.fit_transform(scaled_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Elbow Method for Optimal Clusters**")
                
                # Calculate elbow curve
                if st.button("Calculate Optimal Clusters"):
                    inertias = []
                    k_range = range(1, 11)
                    
                    for k in k_range:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(normalized_data)
                        inertias.append(kmeans.inertia_)
                    
                    fig_elbow = px.line(
                        x=list(k_range),
                        y=inertias,
                        markers=True,
                        title="Elbow Method for Optimal k",
                        labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'}
                    )
                    st.plotly_chart(fig_elbow, use_container_width=True)
            
            with col2:
                st.write("**K-Means Clustering**")
                
                n_clusters = st.slider("Select number of clusters:", 2, 8, 3)
                
                if st.button("Perform Clustering"):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(normalized_data)
                    
                    # Add clusters to original data
                    clustered_data = clustering_data.copy()
                    clustered_data['cluster'] = clusters
                    
                    # Cluster distribution
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    
                    fig_pie = px.pie(
                        values=cluster_counts.values,
                        names=[f'Cluster {i}' for i in cluster_counts.index],
                        title='Cluster Distribution'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Cluster statistics
                    st.write("**Cluster Statistics:**")
                    cluster_stats = clustered_data.groupby('cluster').agg({
                        'sessions_frequency': ['mean', 'count'],
                        'duration': 'mean',
                        'total_traffic': 'mean' if engagement_metrics['total_traffic'].sum() > 0 else lambda x: 0
                    }).round(2)
                    st.dataframe(cluster_stats, use_container_width=True)
            
            # Cluster Visualization
            if 'clustered_data' in locals():
                st.write("**Cluster Visualization**")
                
                fig_scatter = px.scatter(
                    clustered_data,
                    x='duration',
                    y='sessions_frequency' if engagement_metrics['total_traffic'].sum() == 0 else 'total_traffic',
                    color=[f'Cluster {c}' for c in clusters],
                    title='User Engagement Clusters',
                    labels={
                        'duration': 'Duration',
                        'sessions_frequency': 'Sessions Frequency',
                        'total_traffic': 'Total Traffic'
                    }
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Detailed cluster analysis
                st.write("**Detailed Cluster Analysis**")
                for i in range(n_clusters):
                    with st.expander(f"Cluster {i} Analysis"):
                        cluster_data = clustered_data[clustered_data['cluster'] == i]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Users", len(cluster_data))
                        with col2:
                            st.metric("Avg Sessions", f"{cluster_data['sessions_frequency'].mean():.2f}")
                        with col3:
                            if engagement_metrics['total_traffic'].sum() > 0:
                                st.metric("Avg Traffic", f"{cluster_data['total_traffic'].mean():.2f}")
                            else:
                                st.metric("Avg Duration", f"{cluster_data['duration'].mean():.2f}")
                        
                        st.dataframe(cluster_data.describe(), use_container_width=True)
            
            st.markdown("---")
            
            # Application Usage Analysis
            st.subheader("📱 Application Usage Analysis")
            
            app_columns = ['social_media', 'google', 'email', 'youtube', 'netflix', 'gaming']
            
            # Check if app data columns exist (with different naming patterns)
            available_app_cols = []
            for app in app_columns:
                if app in df_renamed.columns:
                    available_app_cols.append(app)
                elif f"{app}_dl_(bytes)" in df.columns and f"{app}_ul_(bytes)" in df.columns:
                    # Calculate total for download + upload
                    df_renamed[app] = df[f"{app}_dl_(bytes)"] + df[f"{app}_ul_(bytes)"]
                    available_app_cols.append(app)
            
            if available_app_cols:
                app_metrics = df_renamed.groupby('msisdn')[available_app_cols].sum()
                
                # Calculate total usage per app
                app_totals = []
                for app in available_app_cols:
                    app_totals.append({'app': app, 'total': app_metrics[app].sum()})
                
                app_total_df = pd.DataFrame(app_totals).sort_values('total', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Total Data Usage by Application**")
                    
                    fig_apps = px.bar(
                        app_total_df,
                        x='total',
                        y='app',
                        orientation='h',
                        title='Total Data Usage by Application',
                        labels={'total': 'Total Data Volume', 'app': 'Application'}
                    )
                    st.plotly_chart(fig_apps, use_container_width=True)
                
                with col2:
                    st.write("**Application Usage Distribution**")
                    
                    fig_pie_apps = px.pie(
                        app_total_df,
                        values='total',
                        names='app',
                        title='Application Usage Distribution'
                    )
                    st.plotly_chart(fig_pie_apps, use_container_width=True)
                
                # Top users per application
                st.write("**Top Users by Application**")
                
                n_top_users = st.slider("Number of top users to show:", 5, 20, 10)
                
                selected_app = st.selectbox("Select application:", available_app_cols)
                
                if selected_app:
                    top_users_app = app_metrics.sort_values(selected_app, ascending=False).head(n_top_users)
                    
                    fig_top_users = px.bar(
                        x=top_users_app.index,
                        y=top_users_app[selected_app],
                        title=f'Top {n_top_users} {selected_app.title()} Users',
                        labels={'x': 'User ID', 'y': f'{selected_app.title()} Usage'}
                    )
                    st.plotly_chart(fig_top_users, use_container_width=True)
            else:
                st.warning("Application usage data not found in the dataset")
            
            # Processing Summary
            st.markdown("---")
            st.subheader("✅ Engagement Analysis Summary")
            
            processing_steps = [
                "✅ Engagement metrics calculated successfully",
                "✅ Top performers identified",
                "✅ Statistical analysis completed",
                "✅ Outlier detection performed",
                "✅ User clustering analysis ready",
                "✅ Application usage patterns analyzed" if available_app_cols else "⚠️ Application usage data not available"
            ]
            
            for step in processing_steps:
                if step.startswith("⚠️"):
                    st.warning(step)
                else:
                    st.write(step)
            
            # Raw engagement data preview
            if st.checkbox("Show Engagement Data Preview"):
                st.subheader("📋 Engagement Data Preview")
                st.dataframe(engagement_metrics.head(100), use_container_width=True)
                
    else:
        st.info("👆 Please upload a CSV file to begin the engagement analysis")
        
        # Show expected data format
        st.subheader("📋 Expected Data Format")
        st.write("Your CSV file should contain columns such as:")
        expected_cols = [
            "msisdn/number", "bearer_id", "dur._(ms)", "total_data",
            "social_media", "google", "email", "youtube", "netflix", "gaming",
            "social_media_dl_(bytes)", "social_media_ul_(bytes)",
            "google_dl_(bytes)", "google_ul_(bytes)", "email_dl_(bytes)", "email_ul_(bytes)",
            "youtube_dl_(bytes)", "youtube_ul_(bytes)", "netflix_dl_(bytes)", "netflix_ul_(bytes)",
            "gaming_dl_(bytes)", "gaming_ul_(bytes)"
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            for i, col in enumerate(expected_cols[:len(expected_cols)//2]):
                st.write(f"• {col}")
        with col2:
            for col in expected_cols[len(expected_cols)//2:]:
                st.write(f"• {col}")


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