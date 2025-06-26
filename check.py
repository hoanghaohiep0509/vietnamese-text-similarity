import importlib
import subprocess
import sys

def check_dependency(package_name):
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def check_java():
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Java is installed")
            return True
        else:
            print("❌ Java is not installed")
            return False
    except:
        print("❌ Java is not installed")
        return False

def main():
    # Kiểm tra Java (cần thiết cho VnCoreNLP)
    java_installed = check_java()
    
    # Kiểm tra các thư viện Python chính
    dependencies = [
        'flask',
        'pandas',
        'numpy',
        'sklearn',
        'underthesea',
        'pyvi',
        'vncorenlp',
        'transformers',
        'torch',
        'gensim',
        'fasttext',
        'sentence_transformers'
    ]
    
    all_installed = all(check_dependency(dep) for dep in dependencies)
    
    if not all_installed or not java_installed:
        print("\n❌ Some dependencies are missing. Please run:")
        print("pip install -r requirements.txt")
        if not java_installed:
            print("\nPlease install Java:")
            print("macOS: brew install openjdk")
            print("Ubuntu: sudo apt-get install default-jre")
        sys.exit(1)
    else:
        print("\n✅ All dependencies are installed correctly!")

if __name__ == "__main__":
    main()