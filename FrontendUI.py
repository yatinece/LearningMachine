import streamlit as st
import requests
import json
from datetime import datetime

# Base URL for the FastAPI endpoints
API_URL = "http://127.0.0.1:8000"

# Helper function to display error messages
def display_error(response):
    st.error(f"Error {response.status_code}: {response.json().get('detail', response.text)}")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Generate Config", "Upload Config", "Execute Pipeline", "List Configs", "View Template"])

# Home Page with a professional banner image
if page == "Home":
    st.title("LearnAI: Ready")
    st.subheader("Automated Data Science Project")
    # Display a banner image. Change the URL below to your professional project image.
    st.image("./learnai.png", use_column_width=True)
    st.markdown("""
    Welcome to the LearnAI: Ready UI. This platform allows you to:
    - **Generate configuration files** for your ML experiments.
    - **Upload and validate** your YAML configuration files.
    - **Execute your ML pipelines** with a single click.
    - **List and download** available configurations.
    
    Use the sidebar to navigate between actions.
    """)

# Generate Config Page
elif page == "Generate Config":
    st.title("Generate YAML Config")
    st.markdown("Provide configuration details in JSON format below. If you’re not sure, leave empty to use defaults.")

    # Simple text area to let user enter a JSON payload; you can expand this with proper fields.
    json_input = st.text_area("Enter configuration JSON", height=250, value='{\n    "data_source": {"type": "ceph", "file_name": "my_data.csv"},\n    "model_settings": {"target_variable": "churn"},\n    "model": {"type": "xgboost"},\n    "hyperparameter_tuning": {"enabled": true},\n    "training": {},\n    "experiment_tracking": {"enabled": true},\n    "output": {}\n}')
    
    if st.button("Generate Config"):
        try:
            config_data = json.loads(json_input)
        except Exception as e:
            st.error(f"Invalid JSON input: {e}")
        else:
            response = requests.post(f"{API_URL}/api/config/generate", json=config_data)
            if response.ok:
                result = response.json()
                st.success("Configuration generated successfully!")
                st.json(result)
            else:
                display_error(response)

# Upload Config Page
elif page == "Upload Config":
    st.title("Upload YAML Configuration File")
    uploaded_file = st.file_uploader("Choose a YAML file", type=["yaml", "yml"])
    
    if uploaded_file is not None:
        if st.button("Upload Config"):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{API_URL}/api/config/upload", files={"file": uploaded_file})
            if response.ok:
                result = response.json()
                st.success("Configuration uploaded successfully!")
                st.json(result)
            else:
                display_error(response)

# Execute Pipeline Page
elif page == "Execute Pipeline":
    st.title("Execute ML Pipeline")
    st.markdown("Select a configuration to run your ML pipeline tasks.")
    
    # List available configs from the API
    response = requests.get(f"{API_URL}/api/config/list")
    if response.ok:
        configs = response.json().get("configs", [])
        if configs:
            config_options = {f"{cfg['filename']} (Created: {cfg['created']})": cfg["filename"] for cfg in configs}
            selected_config = st.selectbox("Select a configuration file", list(config_options.keys()))
            
            # Optional: allow user to enter specific tasks (comma-separated)
            tasks_input = st.text_input("Tasks to execute (optional, comma separated, leave empty to run all tasks)")
            tasks = [task.strip() for task in tasks_input.split(",")] if tasks_input else None

            if st.button("Execute"):
                # The API expects the config_file (the filename) and an optional list of tasks
                payload = {
                    "config_file": config_options[selected_config],
                    "tasks": tasks
                }
                exec_response = requests.post(f"{API_URL}/api/execute", json=payload)
                if exec_response.ok:
                    result = exec_response.json()
                    st.success("Execution request sent successfully!")
                    st.json(result)
                else:
                    display_error(exec_response)
        else:
            st.info("No configuration files found. Please generate or upload a config first.")
    else:
        display_error(response)

# List Configs Page
elif page == "List Configs":
    st.title("List Available Configurations")
    response = requests.get(f"{API_URL}/api/config/list")
    if response.ok:
        configs = response.json().get("configs", [])
        if configs:
            for cfg in configs:
                st.markdown(f"**Filename:** {cfg['filename']}")
                st.markdown(f"- **Path:** {cfg['path']}")
                st.markdown(f"- **Created:** {cfg['created']}")
                st.markdown("---")
        else:
            st.info("No configuration files available.")
    else:
        display_error(response)

# View Template Page
elif page == "View Template":
    st.title("View Config Template")
    st.markdown("Enter a section name to view its template configuration. Examples: `data_source`, `model`, `training`.")

    section = st.text_input("Template Section", value="data_source")
    if st.button("Get Template"):
        response = requests.get(f"{API_URL}/api/config/templates/{section}")
        if response.ok:
            template = response.json()
            st.success("Template retrieved successfully!")
            st.json(template)
        else:
            display_error(response)
