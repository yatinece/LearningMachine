from learnai_ready.core.services import ConfigService

def main():
    # Test the import by creating a ConfigService instance
    config_service = ConfigService()
    print("Successfully imported ConfigService!")
    print(f"ConfigService instance created: {config_service}")

if __name__ == "__main__":
    main() 