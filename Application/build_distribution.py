"""
Automated Build Script for Traffic Analysis System
Builds EXE and creates distribution package
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_step(message):
    """Print a formatted step message"""
    print("\n" + "=" * 60)
    print(f"  {message}")
    print("=" * 60)

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        return True
    except ImportError:
        return False

def install_pyinstaller():
    """Install PyInstaller"""
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build_exe():
    """Build the EXE using PyInstaller"""
    print_step("Building EXE")
    
    # Check for spec file
    if not os.path.exists("traffic_analysis.spec"):
        print("ERROR: traffic_analysis.spec not found!")
        return False
    
    # Build
    try:
        subprocess.check_call(["pyinstaller", "traffic_analysis.spec", "--clean"])
        print("✓ Build successful!")
        return True
    except subprocess.CalledProcessError:
        print("✗ Build failed!")
        return False

def create_distribution():
    """Create distribution package"""
    print_step("Creating Distribution Package")
    
    dist_folder = Path("dist/TrafficAnalysis")
    if not dist_folder.exists():
        print("ERROR: Build folder not found!")
        return False
    
    # Create distribution folder
    package_folder = Path("TrafficAnalysis_Distribution")
    if package_folder.exists():
        shutil.rmtree(package_folder)
    package_folder.mkdir()
    
    print("Copying files...")
    
    # Copy EXE and dependencies
    shutil.copytree(dist_folder, package_folder / "TrafficAnalysis")
    
    # Copy documentation
    docs_to_copy = [
        "STUDENT_GUIDE.md",
        "README.md",
        "INSTALL.md",
    ]
    
    for doc in docs_to_copy:
        if os.path.exists(doc):
            shutil.copy(doc, package_folder)
            print(f"  ✓ Copied {doc}")
    
    # Create sample data folder
    sample_folder = package_folder / "sample_data"
    sample_folder.mkdir()
    
    # Create README in package
    readme_content = """# Traffic Analysis System - Student Package

## Quick Start

1. Run TrafficAnalysis\\TrafficAnalysis.exe
2. Click "Dependencies" button to install packages
3. Follow the STUDENT_GUIDE.md for instructions

## Files Included

- TrafficAnalysis/ - Main application
- STUDENT_GUIDE.md - Quick start guide
- README.md - Full documentation
- INSTALL.md - Installation help
- sample_data/ - Put your videos here

## Need Help?

Read STUDENT_GUIDE.md first!

## Requirements

- Windows 10 or later
- Internet connection (for package installation)
- 5+ GB free space

Good luck with your project!
"""
    
    with open(package_folder / "START_HERE.txt", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✓ Distribution package created: {package_folder}")
    
    # Get size
    total_size = sum(f.stat().st_size for f in package_folder.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    print(f"  Package size: {size_mb:.1f} MB")
    
    return True

def create_zip():
    """Create ZIP file for distribution"""
    print_step("Creating ZIP File")
    
    package_folder = Path("TrafficAnalysis_Distribution")
    if not package_folder.exists():
        print("ERROR: Distribution folder not found!")
        return False
    
    import zipfile
    
    zip_name = "TrafficAnalysis_v1.0.zip"
    print(f"Creating {zip_name}...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in package_folder.rglob('*'):
            if file.is_file():
                arcname = file.relative_to(package_folder.parent)
                zipf.write(file, arcname)
                if file.stat().st_size > 1024 * 1024:  # Only print large files
                    print(f"  Adding {arcname.name} ({file.stat().st_size / (1024*1024):.1f} MB)")
    
    zip_size = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"\n✓ ZIP created: {zip_name} ({zip_size:.1f} MB)")
    
    return True

def main():
    """Main build process"""
    print("""
╔══════════════════════════════════════════════════════════╗
║   Traffic Analysis System - Automated Build Script      ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check PyInstaller
    print_step("Checking Requirements")
    if not check_pyinstaller():
        print("PyInstaller not found. Installing...")
        install_pyinstaller()
    print("✓ PyInstaller available")
    
    # Build EXE
    if not build_exe():
        print("\n✗ Build failed. Please check errors above.")
        return 1
    
    # Create distribution
    if not create_distribution():
        print("\n✗ Distribution creation failed.")
        return 1
    
    # Create ZIP
    if not create_zip():
        print("\n✗ ZIP creation failed.")
        return 1
    
    print_step("BUILD COMPLETE!")
    print("""
Distribution package ready:
  * TrafficAnalysis_v1.0.zip

Next steps:
  1. Test the EXE on a clean machine
  2. Upload ZIP to your course platform
  3. Send STUDENT_GUIDE.md to students
  
Students will:
  1. Extract ZIP
  2. Run TrafficAnalysis.exe
  3. Click Dependencies button
  4. Start analyzing traffic!
    """)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nBuild cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)