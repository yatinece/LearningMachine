import configparser
import os


def create_config():
    config = configparser.ConfigParser()

    # Add sections and key-value pairs
    config['General'] = {'debug': True, 
                         'log_level': 'info' , 
                          'log_file_path':r".\logs\executor_api_logs\executor_api.log", 
                          "project_name"  : 'learn_test' ,
                          "DATABASE_URL" : "sqlite:///./project_management.db" , 
                          "optuna_url" : "sqlite:///./optuna_studies.db" , }
    ensure_file_path(config['General']['log_file_path'])

    # Write the configuration to a file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


def ensure_file_path(file_path="./api.log"):
    # Extract the directory path from the file

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.isfile(file_path):
        open(file_path, 'w').close()
    return file_path

if __name__ == "__main__":
   
    print(create_config())
