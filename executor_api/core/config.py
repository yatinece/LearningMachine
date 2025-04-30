import configparser
import os


def create_config():
    config = configparser.ConfigParser()

    # Add sections and key-value pairs
    config['General'] = {'debug': True, 
                         'log_level': 'info' , 
                          'log_file_path':".\logs\executor_api_logs\executor_api.log",}
    ensure_file_path(config['General']['log_file_path'])
    config['Database'] = {'db_name': 'example_db',
                          'db_host': 'localhost', 'db_port': '5432'}
    config['Project'] = {'user_name': 'admin',
                          'project_name': 'test_build'}

    # Write the configuration to a file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


def create_file_log(log_file="api.log" , log_file_path="./"):
    os,


import os

def ensure_file_path(file_path="./api.log"):
    # Extract the directory path from the file

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.isfile(file_path):
        open(file_path, 'w').close()
    return file_path

if __name__ == "__main__":
   
    print(create_config())
