import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
import subprocess
from pathlib import Path
import json

class SubtitleConfigUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Subtitle Generator & Translator")
        self.root.geometry("800x700")
        
        # Default configuration
        self.config = {
            "target_language": "en",
            "font_size": 20,
            "font_name": "Arial",
            "primary_color": "#FFFFFF",  # White
            "outline_color": "#E361F7",  # Pink
            "back_color": "#000000",     # Black
            "outline_width": 2,
            "shadow": 1,
            "border_style": 3
        }
        
        self.setup_ui()
        self.update_preview()
    
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video selection
        ttk.Label(main_frame, text="Video File:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.video_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.video_path_var, width=50).grid(row=0, column=1, columnspan=2, pady=5)
        ttk.Button(main_frame, text="Browse...", command=self.browse_video).grid(row=0, column=3, pady=5)
        
        # Language selection
        ttk.Label(main_frame, text="Target Language:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.languages = {
            "Arabic": "ar", "German": "de", "Spanish": "es", "French": "fr",
            "Hebrew": "he", "Italian": "it", "Japanese": "ja", "Portuguese": "pt",
            "Russian": "ru", "Chinese": "zh", "English": "en", "Persian": "fa"
        }
        self.lang_var = tk.StringVar(value="English")
        lang_combo = ttk.Combobox(main_frame, textvariable=self.lang_var, values=list(self.languages.keys()), state="readonly", width=20)
        lang_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        lang_combo.bind("<<ComboboxSelected>>", lambda e: self.update_config("target_language", self.languages[self.lang_var.get()]))
        
        # Font settings
        ttk.Separator(main_frame, orient='horizontal').grid(row=2, column=0, columnspan=4, sticky='ew', pady=10)
        ttk.Label(main_frame, text="Font Settings", font=("Arial", 12, "bold")).grid(row=3, column=0, columnspan=4, sticky=tk.W)
        
        # Font family
        ttk.Label(main_frame, text="Font Family:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.font_families = ["Arial", "Times New Roman", "Courier New", "Verdana", "Georgia", "Comic Sans MS", "Impact", "Amiri"]
        self.font_var = tk.StringVar(value="Arial")
        font_combo = ttk.Combobox(main_frame, textvariable=self.font_var, values=self.font_families, state="readonly", width=20)
        font_combo.grid(row=4, column=1, sticky=tk.W, pady=5)
        font_combo.bind("<<ComboboxSelected>>", lambda e: self.update_preview())
        
        # Font size
        ttk.Label(main_frame, text="Font Size:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.font_size_var = tk.IntVar(value=20)
        font_size_scale = ttk.Scale(main_frame, from_=10, to=50, orient=tk.HORIZONTAL, variable=self.font_size_var, command=lambda v: self.update_preview())
        font_size_scale.grid(row=5, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.font_size_label = ttk.Label(main_frame, text="20")
        self.font_size_label.grid(row=5, column=3, pady=5)
        
        # Color settings
        ttk.Separator(main_frame, orient='horizontal').grid(row=6, column=0, columnspan=4, sticky='ew', pady=10)
        ttk.Label(main_frame, text="Color Settings", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=4, sticky=tk.W)
        
        # Primary color (text)
        ttk.Label(main_frame, text="Text Color:").grid(row=8, column=0, sticky=tk.W, pady=5)
        self.primary_color_btn = tk.Button(main_frame, bg="#FFFFFF", width=10, command=lambda: self.choose_color("primary_color"))
        self.primary_color_btn.grid(row=8, column=1, sticky=tk.W, pady=5)
        
        # Text outline color
        ttk.Label(main_frame, text="Text Outline Color:").grid(row=9, column=0, sticky=tk.W, pady=5)
        self.outline_color_btn = tk.Button(main_frame, bg="#E361F7", width=10, command=lambda: self.choose_color("outline_color"))
        self.outline_color_btn.grid(row=9, column=1, sticky=tk.W, pady=5)
        
        # Background color
        ttk.Label(main_frame, text="Background Color:").grid(row=10, column=0, sticky=tk.W, pady=5)
        self.back_color_btn = tk.Button(main_frame, bg="#000000", width=10, command=lambda: self.choose_color("back_color"))
        self.back_color_btn.grid(row=10, column=1, sticky=tk.W, pady=5)
        
        # Outline width
        ttk.Label(main_frame, text="Outline Width:").grid(row=11, column=0, sticky=tk.W, pady=5)
        self.outline_width_var = tk.IntVar(value=2)
        outline_scale = ttk.Scale(main_frame, from_=0, to=5, orient=tk.HORIZONTAL, variable=self.outline_width_var, command=lambda v: self.update_preview())
        outline_scale.grid(row=11, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.outline_width_label = ttk.Label(main_frame, text="2")
        self.outline_width_label.grid(row=11, column=3, pady=5)
        
        # Shadow
        ttk.Label(main_frame, text="Shadow:").grid(row=12, column=0, sticky=tk.W, pady=5)
        self.shadow_var = tk.IntVar(value=1)
        shadow_scale = ttk.Scale(main_frame, from_=0, to=5, orient=tk.HORIZONTAL, variable=self.shadow_var, command=lambda v: self.update_preview())
        shadow_scale.grid(row=12, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.shadow_label = ttk.Label(main_frame, text="1")
        self.shadow_label.grid(row=12, column=3, pady=5)
        
        # Border style
        ttk.Label(main_frame, text="Border Style:").grid(row=13, column=0, sticky=tk.W, pady=5)
        self.border_styles = {
            "Standard Outline (1)": 1,
            "Opaque Box (3)": 3,
            "Outline + Box (4)": 4
        }
        self.border_style_var = tk.StringVar(value="Opaque Box (3)")
        border_combo = ttk.Combobox(main_frame, textvariable=self.border_style_var, values=list(self.border_styles.keys()), state="readonly", width=20)
        border_combo.grid(row=13, column=1, sticky=tk.W, pady=5)
        border_combo.bind("<<ComboboxSelected>>", lambda e: self.update_preview())
        
        # Preview section
        ttk.Separator(main_frame, orient='horizontal').grid(row=14, column=0, columnspan=4, sticky='ew', pady=10)
        ttk.Label(main_frame, text="Preview", font=("Arial", 12, "bold")).grid(row=15, column=0, columnspan=4, sticky=tk.W)
        
        self.preview_canvas = tk.Canvas(main_frame, width=400, height=100, bg="white")
        self.preview_canvas.grid(row=16, column=0, columnspan=4, pady=10)
        
        # Preview text display
        self.preview_text = ttk.Label(main_frame, text="", font=("Arial", 10))
        self.preview_text.grid(row=17, column=0, columnspan=4, pady=5)
        
        # Action buttons
        ttk.Separator(main_frame, orient='horizontal').grid(row=18, column=0, columnspan=4, sticky='ew', pady=10)
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=19, column=0, columnspan=4)
        
        ttk.Button(button_frame, text="Generate Subtitles", command=self.generate_subtitles).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
    
    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.video_path_var.set(filename)
    
    def choose_color(self, color_type):
        color = colorchooser.askcolor(title=f"Choose {color_type.replace('_', ' ').title()}")
        if color[1]:
            self.config[color_type] = color[1]
            if color_type == "primary_color":
                self.primary_color_btn.config(bg=color[1])
            elif color_type == "outline_color":
                self.outline_color_btn.config(bg=color[1])
            elif color_type == "back_color":
                self.back_color_btn.config(bg=color[1])
            self.update_preview()
    
    def update_config(self, key, value):
        self.config[key] = value
        self.update_preview()
    
    def hex_to_ass_color(self, hex_color):
        """Convert web hex color (#RRGGBB) to ASS subtitle format (&HBBGGRR&)"""
        hex_color = hex_color.lstrip('#')
        r = hex_color[0:2]
        g = hex_color[2:4]
        b = hex_color[4:6]
        return f"&H{b.upper()}{g.upper()}{r.upper()}&"
    
    def update_preview(self):
        # Update slider labels
        self.font_size_label.config(text=str(int(self.font_size_var.get())))
        self.outline_width_label.config(text=str(int(self.outline_width_var.get())))
        self.shadow_label.config(text=str(int(self.shadow_var.get())))
        
        # Update config
        self.config["font_size"] = int(self.font_size_var.get())
        self.config["font_name"] = self.font_var.get()
        self.config["outline_width"] = int(self.outline_width_var.get())
        self.config["shadow"] = int(self.shadow_var.get())
        self.config["border_style"] = self.border_styles[self.border_style_var.get()]
        
        # Draw preview on canvas
        self.preview_canvas.delete("all")
        
        # Background color
        bg_color = self.config["back_color"]
        self.preview_canvas.create_rectangle(50, 20, 350, 80, fill=bg_color, outline="")
        
        # Preview text with simulated outline effect
        text = "ABC"
        x, y = 200, 50
        font_size = min(40, int(self.config["font_size"] * 1.5))
        
        try:
            font = (self.config["font_name"], font_size, "bold")
        except:
            font = ("Arial", font_size, "bold")
        
        # Draw outline (simplified - multiple offset texts)
        outline_color = self.config["outline_color"]
        outline_width = self.config["outline_width"]
        
        if outline_width > 0:
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        self.preview_canvas.create_text(x + dx, y + dy, text=text, font=font, fill=outline_color)
        
        # Draw main text
        text_color = self.config["primary_color"]
        self.preview_canvas.create_text(x, y, text=text, font=font, fill=text_color)
        
        # Update preview text info
        primary_ass = self.hex_to_ass_color(self.config["primary_color"])
        outline_ass = self.hex_to_ass_color(self.config["outline_color"])  # Text outline/stroke
        back_ass = self.hex_to_ass_color(self.config["back_color"])  # Background box
        
        preview_info = (f"FontSize={self.config['font_size']}, "
                       f"PrimaryColour={primary_ass}, "
                       f"OutlineColour={outline_ass}, "
                       f"BackColour={back_ass}, "
                       f"Outline={self.config['outline_width']}, "
                       f"Shadow={self.config['shadow']}, "
                       f"BorderStyle={self.config['border_style']}")
        self.preview_text.config(text=preview_info)
    
    def save_config(self):
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.config, f, indent=4)
            messagebox.showinfo("Success", "Configuration saved successfully!")
    
    def load_config(self):
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'r') as f:
                self.config = json.load(f)
            
            # Update UI elements
            self.font_size_var.set(self.config["font_size"])
            self.font_var.set(self.config["font_name"])
            self.outline_width_var.set(self.config["outline_width"])
            self.shadow_var.set(self.config["shadow"])
            
            # Find border style name
            for name, value in self.border_styles.items():
                if value == self.config["border_style"]:
                    self.border_style_var.set(name)
                    break
            
            # Find language name
            for name, code in self.languages.items():
                if code == self.config.get("target_language", "en"):
                    self.lang_var.set(name)
                    break
            
            # Update color buttons
            self.primary_color_btn.config(bg=self.config["primary_color"])
            self.outline_color_btn.config(bg=self.config["outline_color"])
            self.back_color_btn.config(bg=self.config["back_color"])
            
            self.update_preview()
            messagebox.showinfo("Success", "Configuration loaded successfully!")
    
    def generate_subtitles(self):
        video_path = self.video_path_var.get()
        if not video_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return
        
        if not Path(video_path).exists():
            messagebox.showerror("Error", "Video file does not exist!")
            return
        
        # Build command with custom config
        script_path = Path(__file__).parent / "video_subtitles_translator.py"
        
        # Create temporary config file
        config_file = Path(__file__).parent / "temp_subtitle_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f)
        
        # Build command
        cmd = [
            "python",
            str(script_path),
            video_path,
            "--target-language", self.config["target_language"],
            "--config", str(config_file)
        ]
        
        messagebox.showinfo("Processing", f"Starting subtitle generation...\n\nCommand: {' '.join(cmd)}\n\nCheck the terminal for progress.")
        
        try:
            # Run in background
            subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start process:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SubtitleConfigUI(root)
    root.mainloop()
