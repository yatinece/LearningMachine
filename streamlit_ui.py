import streamlit as st
import requests
import pandas as pd
import json
import time
from typing import Dict, Any, List, Optional
import uuid

# Configure the API base URL
API_BASE_URL = "http://localhost:8000"

# Set page configuration
st.set_page_config(
    page_title="ML Experiment Runner",
    page_icon="🧪",
    layout="wide"
)

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "project_id" not in st.session_state:
    st.session_state.project_id = None
if "experiment_id" not in st.session_state:
    st.session_state.experiment_id = None
if "api_responses" not in st.session_state:
    st.session_state.api_responses = {}
if "error_message" not in st.session_state:
    st.session_state.error_message = None


def api_request(method: str, endpoint: str, data: Dict = None, params: Dict = None) -> Dict:
    """Make an API request and handle errors"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        st.session_state.error_message = None
        
        if method.lower() == "get":
            response = requests.get(url, params=params)
        elif method.lower() == "post":
            response = requests.post(url, json=data)
        elif method.lower() == "put":
            response = requests.put(url, json=data)
        elif method.lower() == "patch":
            response = requests.patch(url, json=data)
        else:
            st.error(f"Unsupported HTTP method: {method}")
            return None
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        error_msg = f"API Error: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                if 'detail' in error_details:
                    error_msg = f"API Error: {error_details['detail']}"
            except:
                pass
        
        st.session_state.error_message = error_msg
        return None


def create_user(user_id: str, username: str, email: str, metadata: Dict) -> Dict:
    """Create a new user"""
    data = {
        "user_id": user_id,
        "username": username,
        "email": email,
        "meta_data": metadata
    }
    
    response = api_request("post", "/users/", data=data)
    if response:
        st.session_state.user_id = response["user_id"]
        st.session_state.api_responses["create_user"] = response
    return response


def create_project(user_id: str, name: str, description: str, config: Dict) -> Dict:
    """Create a new project for a user"""
    data = {
        "name": name,
        "description": description,
        "config": config
    }
    
    response = api_request("post", f"/users/{user_id}/projects/", data=data)
    if response:
        st.session_state.project_id = response["id"]
        st.session_state.api_responses["create_project"] = response
    return response


def create_experiment(project_id: str, name: str, description: str, 
                     data_source: str, data_format: str, 
                     target_column: str, task_type: str) -> Dict:
    """Create a new ML experiment for a project"""
    data = {
        "name": name,
        "description": description,
        "data_source": data_source,
        "data_format": data_format,
        "target_column": target_column,
        "task_type": task_type
    }
    
    response = api_request("post", f"/ml/projects/{project_id}/experiments/", data=data)
    if response:
        st.session_state.experiment_id = response["id"]
        st.session_state.api_responses["create_experiment"] = response
    return response


def upload_dataset(experiment_id: str, url: str, file_format: str, options: Dict) -> Dict:
    """Upload a dataset from URL to an experiment"""
    data = {
        "url": url,
        "file_format": file_format,
        "options": options
    }
    
    response = api_request("post", f"/ml/experiments/{experiment_id}/upload-dataset", data=data)
    if response:
        st.session_state.api_responses["upload_dataset"] = response
    return response


def detect_features(experiment_id: str) -> List[Dict]:
    """Detect features in the dataset"""
    response = api_request("post", f"/ml/experiments/{experiment_id}/detect-features")
    if response:
        st.session_state.api_responses["detect_features"] = response
    return response


def train_model(experiment_id: str, optimization_metric: str = "accuracy", 
                n_trials: int = 20, use_gpu: bool = False) -> Dict:
    """Train a model on the dataset"""
    data = {
        "experiment_id": experiment_id,
        "optimization_metric": optimization_metric,
        "n_trials": n_trials,
        "use_gpu": use_gpu
    }
    
    response = api_request("post", f"/ml/experiments/{experiment_id}/train", data=data)
    if response:
        st.session_state.api_responses["train_model"] = response
    return response


def get_experiment_details(experiment_id: str) -> Dict:
    """Get the details of an experiment"""
    response = api_request("get", f"/ml/experiments/{experiment_id}")
    if response:
        st.session_state.api_responses["experiment_details"] = response
    return response


# App header
st.title("🧪 ML Experiment Runner")
st.markdown("A streamlined interface for running ML experiments through the API")

# Display any error messages
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# App tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Create User", 
    "2. Create Project", 
    "3. Create Experiment", 
    "4. Upload Dataset", 
    "5. Train Model",
    "6. Results"
])

# Tab 1: Create User
with tab1:
    st.header("Create User")
    
    with st.form("user_form"):
        user_id = st.text_input("User ID", "E000000")
        username = st.text_input("Username", "data_scientist")
        email = st.text_input("Email", "data_scientist@example.com")
        
        st.subheader("Metadata")
        col1, col2 = st.columns(2)
        with col1:
            org = st.text_input("Organization", "Research Lab")
        with col2:
            role = st.text_input("Role", "Senior Data Scientist")
            
        submitted = st.form_submit_button("Create User")
        
        if submitted:
            metadata = {"organization": org, "role": role}
            with st.spinner("Creating user..."):
                result = create_user(user_id, username, email, metadata)
                if result:
                    st.success(f"User created successfully with ID: {st.session_state.user_id}")
    
    # Display current user
    if st.session_state.user_id:
        st.info(f"Current User ID: {st.session_state.user_id}")
        
        # Option to use existing user
        if st.button("Use existing user"):
            st.success("Using existing user for next steps")

# Tab 2: Create Project
with tab2:
    st.header("Create Project")
    
    if not st.session_state.user_id:
        st.warning("Please create a user first (Step 1)")
    else:
        with st.form("project_form"):
            project_name = st.text_input("Project Name", "Customer Churn Analysis")
            project_desc = st.text_area("Description", "Predict customer churn based on historical data")
            
            st.subheader("Project Configuration")
            goal = st.selectbox("Goal", ["classification", "regression"])
            metrics = st.multiselect("Metrics", 
                                     ["accuracy", "precision", "recall", "f1_score", "roc_auc", "rmse", "mae", "r2"],
                                     default=["accuracy", "f1_score", "roc_auc"])
            
            project_submitted = st.form_submit_button("Create Project")
            
            if project_submitted:
                config = {"goal": goal, "metrics": metrics}
                with st.spinner("Creating project..."):
                    result = create_project(st.session_state.user_id, project_name, project_desc, config)
                    if result:
                        st.success(f"Project created successfully with ID: {st.session_state.project_id}")
    
    # Display current project
    if st.session_state.project_id:
        st.info(f"Current Project ID: {st.session_state.project_id}")
        
        # Option to use existing project
        if st.button("Use existing project"):
            st.success("Using existing project for next steps")

# Tab 3: Create Experiment
with tab3:
    st.header("Create ML Experiment")
    
    if not st.session_state.project_id:
        st.warning("Please create a project first (Step 2)")
    else:
        with st.form("experiment_form"):
            exp_name = st.text_input("Experiment Name", "Initial Churn Model")
            exp_desc = st.text_area("Description", "XGBoost model to predict customer churn")
            
            data_source = st.text_input(
                "Data Source", 
                "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            )
            
            data_format = st.selectbox("Data Format", ["csv", "parquet", "json"])
            target_column = st.text_input("Target Column", "Churn")
            task_type = st.selectbox("Task Type", ["classification", "regression"])
            
            exp_submitted = st.form_submit_button("Create Experiment")
            
            if exp_submitted:
                with st.spinner("Creating experiment..."):
                    result = create_experiment(
                        st.session_state.project_id, exp_name, exp_desc, 
                        data_source, data_format, target_column, task_type
                    )
                    if result:
                        st.success(f"Experiment created successfully with ID: {st.session_state.experiment_id}")
    
    # Display current experiment
    if st.session_state.experiment_id:
        st.info(f"Current Experiment ID: {st.session_state.experiment_id}")
        
        # Option to use existing experiment
        if st.button("Use existing experiment"):
            st.success("Using existing experiment for next steps")

# Tab 4: Upload Dataset
with tab4:
    st.header("Upload Dataset")
    
    if not st.session_state.experiment_id:
        st.warning("Please create an experiment first (Step 3)")
    else:
        with st.form("dataset_form"):
            dataset_url = st.text_input(
                "Dataset URL", 
                "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            )
            file_format = st.selectbox("File Format", ["csv", "parquet", "json"])
            
            st.subheader("Options")
            col1, col2 = st.columns(2)
            with col1:
                separator = st.selectbox("Separator", [",", ";", "\t", "|"])
                header = st.selectbox("Header Row", ['infer', 'int'])
            with col2:
                index_col = st.text_input("Index Column", "0")
                if index_col == "":
                    index_col = None
                else:
                    try:
                        index_col = int(index_col)
                    except:
                        pass
            
            dataset_submitted = st.form_submit_button("Upload Dataset")
            
            if dataset_submitted:
                options = {
                    "sep": separator,
                    "header": header,
                    "index_col": index_col
                }
                
                with st.spinner("Uploading dataset..."):
                    result = upload_dataset(st.session_state.experiment_id, dataset_url, file_format, options)
                    if result:
                        st.success("Dataset uploaded successfully")
                        
                        # Auto-detect features
                        with st.spinner("Detecting features..."):
                            features = detect_features(st.session_state.experiment_id)
                            if features:
                                st.success(f"Detected {len(features)} features")
                                
                                # Display detected features
                                st.subheader("Detected Features")
                                feature_df = pd.DataFrame(features)
                                st.dataframe(feature_df)

# Tab 5: Train Model
with tab5:
    st.header("Train Model")
    
    if not st.session_state.experiment_id:
        st.warning("Please create an experiment first (Step 3)")
    else:
        with st.form("training_form"):
            st.subheader("Training Configuration")
            col1, col2 = st.columns(2)
            with col1:
                optimization_metric = st.selectbox(
                    "Optimization Metric", 
                    ["accuracy", "precision", "recall", "f1", "roc_auc", "rmse", "mae", "r2"]
                )
                n_trials = st.slider("Number of Trials", 5, 50, 20, 5)
            with col2:
                use_gpu = st.checkbox("Use GPU (if available)", False)
            
            train_submitted = st.form_submit_button("Train Model")
            
            if train_submitted:
                with st.spinner("Training model... This may take a while"):
                    result = train_model(
                        st.session_state.experiment_id, 
                        optimization_metric=optimization_metric,
                        n_trials=n_trials,
                        use_gpu=use_gpu
                    )
                    if result:
                        st.success("Model trained successfully!")
                        st.session_state.model_results = result

# Tab 6: Results
with tab6:
    st.header("Experiment Results")
    
    if not st.session_state.experiment_id:
        st.warning("Please create and run an experiment first (Steps 1-5)")
    else:
        if st.button("Refresh Results"):
            with st.spinner("Fetching experiment details..."):
                experiment = get_experiment_details(st.session_state.experiment_id)
        
        # Display API responses
        st.subheader("Complete Workflow Results")
        
        for step, response in st.session_state.api_responses.items():
            with st.expander(f"{step.replace('_', ' ').title()}"):
                st.json(response)
        
        # Show model training results
        if "train_model" in st.session_state.api_responses:
            training_results = st.session_state.api_responses["train_model"]
            
            st.subheader("Model Performance")
            
            # Display metrics
            if "metrics" in training_results:
                metrics = training_results["metrics"]
                metrics_df = pd.DataFrame([metrics])
                st.dataframe(metrics_df)
            
            # Feature importance
            if "feature_importance" in training_results:
                st.subheader("Feature Importance")
                feature_imp = training_results["feature_importance"]
                
                # Convert to dataframe for visualization
                fi_df = pd.DataFrame(
                    [(k, v) for k, v in feature_imp.items()], 
                    columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False)
                
                st.bar_chart(fi_df.set_index("Feature"))
            
            # Best parameters
            if "best_parameters" in training_results:
                st.subheader("Best Parameters")
                st.json(training_results["best_parameters"])

st.markdown("---")
st.caption("ML Experiment Runner v1.0")

# Instructions in the sidebar
with st.sidebar:
    st.title("Instructions")
    st.markdown("""
    1. **Create User** - Create a new user account
    2. **Create Project** - Set up a new ML project
    3. **Create Experiment** - Configure an experiment
    4. **Upload Dataset** - Upload data from a URL
    5. **Train Model** - Train and optimize an XGBoost model
    6. **Results** - View experiment results
    
    Follow the tabs in order to complete a full experiment workflow.
    """)
    
    st.markdown("---")
    
    # Reset session state
    if st.button("Reset All", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# Run the app with: streamlit run app.py