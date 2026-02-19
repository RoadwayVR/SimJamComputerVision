"""
Dependency Manager - Checks and installs required packages
"""

import sys
import subprocess
import importlib
from typing import List, Tuple, Dict
import threading


class DependencyManager:
    """Manages Python package dependencies"""
    
    # Required packages with their import names and pip names
    REQUIRED_PACKAGES = {
        'customtkinter': 'customtkinter>=5.2.0',
        'PIL': 'pillow>=10.0.0',
        'cv2': 'opencv-python>=4.8.0',
        'numpy': 'numpy>=1.24.0',
        'pandas': 'pandas>=2.0.0',
        'ultralytics': 'ultralytics>=8.0.0',
        'supervision': 'supervision>=0.17.0',
    }
    
    def __init__(self):
        self.missing_packages = []
        self.installed_packages = []
        self.status_callback = None
        
    def check_dependencies(self) -> Tuple[List[str], List[str]]:
        """
        Check which dependencies are installed
        
        Returns:
            Tuple of (installed_packages, missing_packages)
        """
        installed = []
        missing = []
        
        for import_name, pip_name in self.REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(import_name)
                installed.append(pip_name.split('>=')[0])
            except ImportError:
                missing.append(pip_name)
        
        self.installed_packages = installed
        self.missing_packages = missing
        
        return installed, missing
    
    def get_status_report(self) -> str:
        """Get a formatted status report"""
        installed, missing = self.check_dependencies()
        
        report = "DEPENDENCY STATUS\n"
        report += "=" * 50 + "\n\n"
        
        if installed:
            report += "✓ INSTALLED:\n"
            for pkg in installed:
                report += f"  • {pkg}\n"
            report += "\n"
        
        if missing:
            report += "✗ MISSING:\n"
            for pkg in missing:
                report += f"  • {pkg}\n"
            report += "\n"
        
        if not missing:
            report += "🎉 All dependencies are installed!\n"
        else:
            report += f"⚠ Need to install {len(missing)} package(s)\n"
        
        return report
    
    def install_package(self, package: str) -> Tuple[bool, str]:
        """
        Install a single package
        
        Args:
            package: Package name with version (e.g., 'numpy>=1.24.0')
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if self.status_callback:
                self.status_callback(f"Installing {package}...")
            
            # Use pip to install
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per package
            )
            
            if result.returncode == 0:
                return True, f"✓ {package} installed successfully"
            else:
                return False, f"✗ Failed to install {package}: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return False, f"✗ Installation timeout for {package}"
        except Exception as e:
            return False, f"✗ Error installing {package}: {str(e)}"
    
    def install_all_missing(self, progress_callback=None) -> Dict[str, Tuple[bool, str]]:
        """
        Install all missing packages
        
        Args:
            progress_callback: Function to call with progress updates
            
        Returns:
            Dictionary of {package: (success, message)}
        """
        self.status_callback = progress_callback
        results = {}
        
        _, missing = self.check_dependencies()
        
        if not missing:
            if progress_callback:
                progress_callback("All packages already installed!")
            return results
        
        total = len(missing)
        for i, package in enumerate(missing, 1):
            if progress_callback:
                progress_callback(f"Installing {i}/{total}: {package}")
            
            success, message = self.install_package(package)
            results[package] = (success, message)
            
            if progress_callback:
                progress_callback(message)
        
        # Final check
        if progress_callback:
            progress_callback("Verifying installation...")
        
        installed, still_missing = self.check_dependencies()
        
        if not still_missing:
            if progress_callback:
                progress_callback("✓ All dependencies installed successfully!")
        else:
            if progress_callback:
                progress_callback(f"⚠ {len(still_missing)} package(s) still missing")
        
        return results
    
    def install_all_missing_async(self, progress_callback=None, completion_callback=None):
        """
        Install all missing packages in a background thread
        
        Args:
            progress_callback: Function to call with progress updates
            completion_callback: Function to call when complete
        """
        def worker():
            results = self.install_all_missing(progress_callback)
            if completion_callback:
                completion_callback(results)
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread


def check_dependencies_cli():
    """Command-line interface for checking dependencies"""
    manager = DependencyManager()
    print(manager.get_status_report())
    
    installed, missing = manager.check_dependencies()
    
    if missing:
        response = input("\nInstall missing packages? (y/n): ").strip().lower()
        if response == 'y':
            print("\nInstalling packages...")
            results = manager.install_all_missing(
                progress_callback=lambda msg: print(f"  {msg}")
            )
            
            print("\n" + "=" * 50)
            print("INSTALLATION COMPLETE")
            print("=" * 50)
            
            success_count = sum(1 for success, _ in results.values() if success)
            fail_count = len(results) - success_count
            
            print(f"✓ Successful: {success_count}")
            print(f"✗ Failed: {fail_count}")
            
            if fail_count > 0:
                print("\nFailed packages:")
                for pkg, (success, msg) in results.items():
                    if not success:
                        print(f"  • {pkg}: {msg}")
    else:
        print("\n✓ Ready to run the application!")
    
    return len(missing) == 0


if __name__ == "__main__":
    check_dependencies_cli()
