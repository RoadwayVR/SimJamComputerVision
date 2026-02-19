"""
Traffic Analysis Application - Main GUI
Modern interface for vehicle detection and intersection performance analytics
"""

import os
import sys
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image

# Import dependency manager
from dependency_manager import DependencyManager

# Import the detection and analytics modules
import detection_module
import analytics_module


class TrafficAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("SimJamComputerVisionAnalytics")
        self.geometry("1200x800")
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Create main container
        self.main_container = ctk.CTkFrame(self, corner_radius=0)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1)
        
        # Create header
        self.create_header()
        
        # Create tabview
        self.tabview = ctk.CTkTabview(self.main_container, corner_radius=10)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))
        
        # Add tabs
        self.tabview.add("Detection & Tracking")
        self.tabview.add("Performance Analytics")
        
        # Configure tab grids
        self.tabview.tab("Detection & Tracking").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Detection & Tracking").grid_rowconfigure(0, weight=1)
        self.tabview.tab("Performance Analytics").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Performance Analytics").grid_rowconfigure(0, weight=1)
        
        # Initialize detection module
        self.detection_frame = detection_module.DetectionTab(
            self.tabview.tab("Detection & Tracking")
        )
        self.detection_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Initialize analytics module
        self.analytics_frame = analytics_module.AnalyticsTab(
            self.tabview.tab("Performance Analytics")
        )
        self.analytics_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Status bar
        self.create_status_bar()
        
    def create_header(self):
        """Create application header with title and info"""
        header_frame = ctk.CTkFrame(self.main_container, height=80, corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_columnconfigure(1, weight=1)
        header_frame.grid_propagate(False)
        
        # LEFT SIDE: Logo and Title container
        left_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        left_container.grid(row=0, column=0, padx=30, pady=10, sticky="w", rowspan=2)
        
        # Try to load logo
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RoadwayVR.jpg")
        if os.path.exists(logo_path):
            try:
                logo_image = Image.open(logo_path)
                logo_image = logo_image.resize((50, 50), Image.Resampling.LANCZOS)
                self.logo_photo = ctk.CTkImage(light_image=logo_image, dark_image=logo_image, size=(50, 50))
                logo_label = ctk.CTkLabel(left_container, image=self.logo_photo, text="")
                logo_label.pack(side="left", padx=(0, 15))
            except Exception as e:
                print(f"Could not load logo: {e}")
        
        # Title
        title_label = ctk.CTkLabel(
            left_container,
            text="SimJam ComputerVision Analytics",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(side="left")
        
        # Info button
        info_btn = ctk.CTkButton(
            header_frame,
            text="ℹ About",
            width=100,
            command=self.show_about,
            fg_color="transparent",
            border_width=2
        )
        info_btn.grid(row=0, column=2, rowspan=2, padx=(10, 10), pady=10, sticky="e")
        
        # Dependencies button
        deps_btn = ctk.CTkButton(
            header_frame,
            text="🔧 Dependencies",
            width=130,
            command=self.show_dependencies,
            fg_color=("gray70", "gray30"),
            border_width=2
        )
        deps_btn.grid(row=0, column=3, rowspan=2, padx=(0, 30), pady=10, sticky="e")

    def create_status_bar(self):
        """Create status bar at bottom"""
        self.status_frame = ctk.CTkFrame(self.main_container, height=30, corner_radius=0)
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        self.status_frame.grid_propagate(False)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            font=ctk.CTkFont(size=11)
        )
        self.status_label.pack(side="left", padx=20, pady=5)
        
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.configure(text=message)
        self.update_idletasks()
        
    def show_about(self):
        """Show About dialog"""
        about_window = ctk.CTkToplevel(self)  # Changed from self.root to self
        about_window.title("About")
        about_window.geometry("350x220")
        about_window.resizable(False, False)
        
        # Center the window
        about_window.transient(self)  # Changed from self.root to self
        about_window.grab_set()
        
        # Content frame with tighter padding
        content_frame = ctk.CTkFrame(about_window, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=25, pady=20)
        
        # Title
        title = ctk.CTkLabel(content_frame, text="About", 
                            font=ctk.CTkFont(size=18, weight="bold"))
        title.pack(pady=(0, 12))
        
        # Developed by
        dev_label = ctk.CTkLabel(content_frame, text="Developed by RoadwayVR", 
                                font=ctk.CTkFont(size=13, weight="bold"),
                                text_color="lightblue")
        dev_label.pack(pady=(0, 10))
        
        # Application info
        app_label = ctk.CTkLabel(content_frame, text="Application:", 
                                font=ctk.CTkFont(size=10),
                                text_color="gray60")
        app_label.pack(pady=(0, 2))
        
        app_name = ctk.CTkLabel(content_frame, text="SimJamCV Analytics", 
                            font=ctk.CTkFont(size=12, weight="bold"))
        app_name.pack(pady=(0, 8))
        
        # Version
        version_label = ctk.CTkLabel(content_frame, text="Version: 1.0", 
                                    font=ctk.CTkFont(size=11))
        version_label.pack(pady=(0, 8))
        
        # Description
        desc_label = ctk.CTkLabel(content_frame, 
                                text="Traffic analysis tool for\nvehicle detection and analytics",
                                font=ctk.CTkFont(size=10),
                                text_color="gray60",
                                justify="center")
        desc_label.pack(pady=(0, 12))
        
        # Close button
        close_btn = ctk.CTkButton(content_frame, text="Close", 
                                command=about_window.destroy, width=100, height=28)
        close_btn.pack()
    
    def show_dependencies(self):
        """Show dependency manager dialog"""
        deps_window = ctk.CTkToplevel(self)
        deps_window.title("Dependency Manager")
        deps_window.geometry("600x500")
        deps_window.transient(self)
        deps_window.grab_set()
        
        # Center the window
        deps_window.update_idletasks()
        x = (deps_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (deps_window.winfo_screenheight() // 2) - (500 // 2)
        deps_window.geometry(f"+{x}+{y}")
        
        # Content frame
        content = ctk.CTkFrame(deps_window)
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            content,
            text="📦 Package Dependencies",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=(10, 5))
        
        subtitle = ctk.CTkLabel(
            content,
            text="Check and install required Python packages",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        subtitle.pack(pady=(0, 15))
        
        # Status text
        status_text = ctk.CTkTextbox(
            content,
            wrap="word",
            font=ctk.CTkFont(family="Courier", size=11)
        )
        status_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Progress bar
        progress = ctk.CTkProgressBar(content)
        progress.pack(fill="x", padx=10, pady=5)
        progress.set(0)
        
        # Button frame
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        # Create dependency manager
        dep_manager = DependencyManager()
        
        def update_status(message):
            """Update status text"""
            status_text.insert("end", f"{message}\n")
            status_text.see("end")
            deps_window.update_idletasks()
        
        def check_deps():
            """Check dependencies"""
            status_text.delete("1.0", "end")
            update_status("Checking dependencies...\n")
            report = dep_manager.get_status_report()
            status_text.insert("end", report)
            
            # Update button states
            installed, missing = dep_manager.check_dependencies()
            if missing:
                install_btn.configure(state="normal")
                progress.set(len(installed) / (len(installed) + len(missing)))
            else:
                install_btn.configure(state="disabled")
                progress.set(1.0)
        
        def install_deps():
            """Install missing dependencies"""
            install_btn.configure(state="disabled", text="Installing...")
            check_btn.configure(state="disabled")
            close_btn.configure(state="disabled")
            
            status_text.delete("1.0", "end")
            update_status("Starting installation...\n")
            
            def on_complete(results):
                """Called when installation completes"""
                update_status("\n" + "=" * 50)
                update_status("INSTALLATION COMPLETE")
                update_status("=" * 50 + "\n")
                
                success_count = sum(1 for success, _ in results.values() if success)
                fail_count = len(results) - success_count
                
                update_status(f"✓ Successful: {success_count}")
                update_status(f"✗ Failed: {fail_count}\n")
                
                if fail_count > 0:
                    update_status("Failed packages:")
                    for pkg, (success, msg) in results.items():
                        if not success:
                            update_status(f"  • {pkg}")
                
                # Re-check
                check_deps()
                
                # Re-enable buttons
                install_btn.configure(text="Install Missing")
                check_btn.configure(state="normal")
                close_btn.configure(state="normal")
                
                if fail_count == 0:
                    messagebox.showinfo(
                        "Success",
                        "All dependencies installed successfully!\n\n"
                        "You can now use all features of the application.",
                        parent=deps_window
                    )
            
            # Install in background
            dep_manager.install_all_missing_async(
                progress_callback=update_status,
                completion_callback=on_complete
            )
        
        # Check button
        check_btn = ctk.CTkButton(
            btn_frame,
            text="🔍 Check Status",
            command=check_deps,
            width=150
        )
        check_btn.pack(side="left", padx=5)
        
        # Install button
        install_btn = ctk.CTkButton(
            btn_frame,
            text="⬇ Install Missing",
            command=install_deps,
            width=150,
            fg_color=("green", "darkgreen")
        )
        install_btn.pack(side="left", padx=5)
        
        # Close button
        close_btn = ctk.CTkButton(
            btn_frame,
            text="Close",
            command=deps_window.destroy,
            width=100
        )
        close_btn.pack(side="right", padx=5)
        
        # Auto-check on open
        check_deps()


def main():
    """Main entry point"""
    # Check dependencies first
    print("Checking dependencies...")
    dep_manager = DependencyManager()
    installed, missing = dep_manager.check_dependencies()
    
    if missing:
        print("\n⚠ WARNING: Missing dependencies detected!")
        print(f"Missing packages: {', '.join([p.split('>=')[0] for p in missing])}")
        print("\nThe application will start, but some features may not work.")
        print("Please click the 'Dependencies' button in the app to install them.")
        print("\nPress Enter to continue...")
        input()
    else:
        print("✓ All dependencies installed!")
    
    # Launch application
    app = TrafficAnalysisApp()
    app.mainloop()


if __name__ == "__main__":
    main()
