import subprocess

def install_requirements():
    try:
        subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
        print("All packages from requirements.txt have been successfully installed.")
    except subprocess.CalledProcessError:
        print("Error installing packages from requirements.txt.")

if __name__ == "__main__":
    install_requirements()
