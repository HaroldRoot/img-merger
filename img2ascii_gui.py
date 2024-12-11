import os
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox


class ASCIIArtGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image to ASCII Art")
        self.root.state("zoomed")  # Maximize window

        # Variables
        self.file_path = tk.StringVar()
        self.method = tk.StringVar(value="luminosity")
        self.kmeans = tk.BooleanVar(value=False)
        self.colorful = tk.BooleanVar(value=False)
        self.invert = tk.BooleanVar(value=False)
        self.font_size = tk.IntVar(value=10)
        self.output_text = None

        # Path to Python interpreter in Poetry environment
        self.python_interpreter = os.path.join(os.path.dirname(__file__),
                                               ".venv", "Scripts", "python.exe")

        # Layout
        self.create_widgets()

    def create_widgets(self):
        # File selection
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=5, padx=10, fill="x")

        tk.Label(file_frame, text="Image File:").pack(side="left")
        tk.Entry(file_frame, textvariable=self.file_path, width=40).pack(
            side="left", padx=5)
        tk.Button(file_frame, text="Browse", command=self.browse_file).pack(
            side="left")

        # Options
        options_frame = tk.LabelFrame(self.root, text="Options")
        options_frame.pack(pady=10, padx=10, fill="x")

        tk.Checkbutton(options_frame, text="Use K-means",
                       variable=self.kmeans).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(options_frame, text="Colorful Output",
                       variable=self.colorful).grid(row=0, column=1, sticky="w")
        tk.Checkbutton(options_frame, text="Invert Brightness",
                       variable=self.invert).grid(row=0, column=2, sticky="w")

        tk.Label(options_frame, text="Brightness Method:").grid(row=1, column=0,
                                                                sticky="e")
        ttk.Combobox(options_frame, textvariable=self.method,
                     values=["average", "min_max", "luminosity"],
                     state="readonly").grid(row=1, column=1, sticky="w")

        # Font size
        font_frame = tk.Frame(self.root)
        font_frame.pack(pady=5, padx=10, fill="x")
        tk.Label(font_frame, text="Font Size:").pack(side="left")
        font_spinbox = tk.Spinbox(font_frame, from_=1, to=999,
                                  textvariable=self.font_size, width=5,
                                  command=self.update_font_size)
        font_spinbox.pack(side="left", padx=5)

        # Output area
        output_frame = tk.Frame(self.root)
        output_frame.pack(pady=5, padx=10, fill="both", expand=True)

        self.output_text = tk.Text(output_frame, wrap="none",
                                   font=("Courier", self.font_size.get()),
                                   bg="black", fg="white")
        self.output_text.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(output_frame, command=self.output_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.output_text.config(yscrollcommand=scrollbar.set)

        # Buttons
        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=10, padx=10)

        tk.Button(buttons_frame, text="Generate ASCII",
                  command=self.generate_ascii).pack(side="left", padx=5)
        tk.Button(buttons_frame, text="Clear", command=self.clear_output).pack(
            side="left", padx=5)

        # Font size adjustment
        self.output_text.bind("<Control-plus>", self.increase_font_size)
        self.output_text.bind("<Control-minus>", self.decrease_font_size)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.file_path.set(file_path)

    def generate_ascii(self):
        file_path = self.file_path.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Invalid file path!")
            return

        self.output_text.insert("end", "Generating ASCII art...\n")
        self.output_text.update()

        threading.Thread(target=self.run_ascii_command,
                         args=(file_path,)).start()

    def run_ascii_command(self, file_path):
        try:
            command = [self.python_interpreter, "./img2ascii.py", file_path,
                       "-b", self.method.get().strip()]
            if self.kmeans.get():
                command.append("-k")
            if self.colorful.get():
                command.append("-f")
            if self.invert.get():
                command.append("-i")

            env = os.environ.copy()
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True,
                                       env=env)
            stdout, stderr = process.communicate()

            if stdout:
                self.display_ascii(stdout)
            if stderr:
                self.output_text.insert("end", f"Error: {stderr}\n")

        except Exception as e:
            self.output_text.insert("end", f"Error: {e}\n")

    def display_ascii(self, ascii_art):
        self.output_text.delete("1.0", "end")
        self.output_text.insert("end", ascii_art)

    def clear_output(self):
        self.output_text.delete("1.0", "end")

    def update_font_size(self):
        self.output_text.config(font=("Courier", self.font_size.get()))

    def increase_font_size(self, event):
        self.font_size.set(self.font_size.get() + 1)
        self.update_font_size()

    def decrease_font_size(self, event):
        self.font_size.set(self.font_size.get() - 1)
        self.update_font_size()


if __name__ == "__main__":
    root = tk.Tk()
    app = ASCIIArtGUI(root)
    root.mainloop()
