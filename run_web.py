import os
import sys
import subprocess
from pathlib import Path

def main():
    app_dir = Path(__file__).resolve().parent
    server_path = app_dir / "web" / "server.py"
    
    if not server_path.exists():
        print(f"Error: Web server script not found at {server_path}")
        sys.exit(1)
        
    print("==================================================")
    print("          Launching SubGen Web Interface          ")
    print("==================================================")
    
    try:
        # Run the server script using the current python interpreter
        subprocess.run([sys.executable, str(server_path)], check=True)
    except KeyboardInterrupt:
        print("\nSubGen Web Interface stopped.")
    except Exception as e:
        print(f"Error launching server: {e}")

if __name__ == "__main__":
    main()
