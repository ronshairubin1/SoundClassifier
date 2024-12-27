import os
import sys

# Add the src directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from app import app, Config

def main():
    # Initialize directories
    Config.init_directories()
    
    # Run the Flask application
    app.run(debug=True, port=5000)

if __name__ == "__main__":
    main() 