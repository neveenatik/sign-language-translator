"""
Create necessary directories for data and models.
"""

from pathlib import Path

def initialize_directories():
    """Create all required directories."""
    
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "scripts",
        "notebooks",
        "src",
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    # Create placeholder files to preserve empty directories in git
    placeholder_files = [
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
    ]
    
    for file_path in placeholder_files:
        Path(file_path).touch(exist_ok=True)
        print(f"✓ Created: {file_path}")
    
    print("\n✅ Directory structure initialized!")

if __name__ == "__main__":
    initialize_directories()
