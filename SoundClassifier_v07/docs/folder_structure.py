import os
import re
from datetime import datetime

def list_directory_tree(startpath):
    """
    Create a Markdown file containing the directory tree starting from startpath.
    Files are organized by importance and common development directories are excluded.
    Output is saved in the same directory as this script.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime('%H-%M_%d-%m-%Y')
    output_file = os.path.join(script_dir, f'.directory_tree_{timestamp}.md')
    
    # Directories to ignore completely
    ignore = {
        '__pycache__',
        '.old_files',
        '.test_files',
        'api_test_app',
        '.pytest_cache',
        'node_modules',
        'build',
        'dist',
        'coverage',
        'voice_to_text.egg-info',
        '.AA_coding_utilities',
        '.git'
    }
    
    # Priority order for files and directories
    priority_order = {
        # Top-level files
        'README.md': 1,
        '.env': 2,
        'requirements.txt': 3,
        'setup.py': 4,
        'run.py': 5,
        'Dockerfile': 6,
        'docker-compose.yml': 7,
        '.gitignore': 8,
        
        # Core directories
        'app': 10,
        'api': 11,
        'terminal_apps': 12,
        'config': 13,
        'scripts': 14,
        'utilities': 15,
        
        # Documentation and support
        'AA_README': 20,
        'docs': 21,
        
        # Environment
        'venv': 90,
    }
    
    def get_priority(name):
        """Get priority for sorting. Lower numbers appear first."""
        return priority_order.get(name, 50)  # Default priority for unlisted items
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Project Directory Tree\n\n")
        f.write(f"Generated on: {timestamp}\n\n")
        f.write(f"Root directory: `{os.path.abspath(startpath)}`\n\n")
        f.write("```plaintext\n")
        
        for root, dirs, files in os.walk(startpath):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if d not in ignore]
            
            # Sort directories by priority
            dirs.sort(key=lambda x: (get_priority(x), x.lower()))
            
            # Calculate level for indentation
            level = root.replace(startpath, '').count(os.sep)
            indent = '│   ' * level
            
            # Write directory name
            relative_path = os.path.relpath(root, startpath)
            if relative_path == '.':
                f.write(f"{os.path.basename(os.path.abspath(startpath))}/\n")
            else:
                dirname = os.path.basename(root)
                if dirname == 'venv':
                    f.write(f'{indent}├── venv/ {{Python virtual environment}}\n')
                    dirs[:] = []  # Skip processing venv subdirectories
                    continue
                else:
                    f.write(f'{indent}├── {dirname}/\n')
            
            # Sort and write files
            files = [f for f in files if not f.startswith('.') and not f.endswith('.pyc')]
            sorted_files = sorted(files, key=lambda x: (get_priority(x), x.lower()))
            
            subindent = '│   ' * (level + 1)
            for i, file in enumerate(sorted_files):
                if i == len(sorted_files) - 1:
                    f.write(f'{subindent}└── {file}\n')
                else:
                    f.write(f'{subindent}├── {file}\n')
        
        f.write("```\n")
        f.write("\n## Excluded Directories\n\n")
        f.write("The following directories are excluded from this tree:\n")
        for dir_name in sorted(ignore):
            f.write(f"- `{dir_name}/`\n")
        
        print(f"\nDirectory tree has been written to: {output_file}")

if __name__ == '__main__':
    # Use the current directory as the start path
    startpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    list_directory_tree(startpath)
