import os
import json
import customtkinter as ctk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ---------------- CORE LOGIC ---------------- #

def get_folder_structure(path, exclusions):
    structure = {}

    try:
        items = [
            item for item in os.listdir(path)
            if not item.startswith(".") and item not in exclusions
        ]
    except PermissionError:
        return "Permission Denied"

    for item in sorted(items):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            structure[item] = get_folder_structure(full_path, exclusions)
        else:
            structure[item] = "file"

    return structure


def dict_to_ascii(data, indent=""):
    lines = []
    keys = list(data.keys())

    for i, key in enumerate(keys):
        connector = "└── " if i == len(keys) - 1 else "├── "
        lines.append(f"{indent}{connector}{key}")

        if isinstance(data[key], dict):
            extension = "    " if i == len(keys) - 1 else "│   "
            lines.extend(dict_to_ascii(data[key], indent + extension))

    return lines


def ascii_to_folders(ascii_text, base_path):
    stack = []

    for line in ascii_text.splitlines():
        if not line.strip():
            continue

        depth = line.count("│") + line.count("    ")
        name = line.strip().replace("├── ", "").replace("└── ", "")

        while len(stack) > depth:
            stack.pop()

        path = os.path.join(base_path, *stack, name)

        if "." in name:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "a").close()
        else:
            os.makedirs(path, exist_ok=True)

        stack.append(name)


# ---------------- GUI APP ---------------- #

class FolderStudio(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.title("Folder Structure Studio")
        self.geometry("1000x650")

        self.root_path = None
        self.exclusions = set()
        self.loaded_json = None

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(expand=True, fill="both", padx=20, pady=20)

        self.tab_auto = self.tabview.add("Auto → JSON")
        self.tab_ascii = self.tabview.add("ASCII ↔ JSON")

        self.build_auto_tab()
        self.build_ascii_tab()
        self.build_confirm_button()

    # ---------- TAB 1 ---------- #
    def build_auto_tab(self):
        frame = ctk.CTkFrame(self.tab_auto)
        frame.pack(fill="both", expand=True)

        self.drop_label = ctk.CTkLabel(
            frame, text="Drag & Drop Folder Here", height=40
        )
        self.drop_label.pack(fill="x", padx=10, pady=10)

        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind("<<Drop>>", self.on_drop)

        ctk.CTkButton(
            frame, text="Select Folder", command=self.select_folder
        ).pack(pady=5)

        self.path_label = ctk.CTkLabel(frame, text="No folder selected")
        self.path_label.pack()

        self.exclude_entry = ctk.CTkEntry(
            frame, placeholder_text="Exclude names (comma-separated)"
        )
        self.exclude_entry.pack(fill="x", padx=20, pady=10)

        self.preview_box = ctk.CTkTextbox(frame, height=300)
        self.preview_box.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------- TAB 2 ---------- #
    def build_ascii_tab(self):
        frame = ctk.CTkFrame(self.tab_ascii)
        frame.pack(fill="both", expand=True)

        self.ascii_box = ctk.CTkTextbox(frame, height=300)
        self.ascii_box.pack(fill="both", expand=True, padx=10, pady=10)

        btn_frame = ctk.CTkFrame(frame)
        btn_frame.pack(pady=10)

        ctk.CTkButton(
            btn_frame, text="Load JSON", command=self.load_json
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            btn_frame, text="Generate Folder", command=self.generate_from_ascii
        ).pack(side="left", padx=10)

    # ---------- CONFIRM ---------- #
    def build_confirm_button(self):
        ctk.CTkButton(
            self, text="Confirm", width=140, command=self.confirm
        ).place(x=20, y=600)

    # ---------- ACTIONS ---------- #
    def on_drop(self, event):
        path = event.data.strip("{}")
        if os.path.isdir(path):
            self.set_root(path)

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.set_root(path)

    def set_root(self, path):
        self.root_path = path
        self.path_label.configure(text=path)
        self.update_preview()

    def update_preview(self):
        self.preview_box.delete("0.0", "end")

        exclusions = {
            e.strip() for e in self.exclude_entry.get().split(",") if e.strip()
        }

        structure = get_folder_structure(self.root_path, exclusions)
        ascii_tree = "\n".join(dict_to_ascii(structure))

        self.preview_box.insert("0.0", ascii_tree)

    def confirm(self):
        if not self.root_path:
            messagebox.showerror("Error", "No folder selected")
            return

        exclusions = {
            e.strip() for e in self.exclude_entry.get().split(",") if e.strip()
        }

        structure = get_folder_structure(self.root_path, exclusions)
        output = os.path.join(self.root_path, "project_structure.json")

        with open(output, "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=4)

        messagebox.showinfo("Success", f"Saved:\n{output}")

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ascii_tree = "\n".join(dict_to_ascii(data))
        self.ascii_box.delete("0.0", "end")
        self.ascii_box.insert("0.0", ascii_tree)

    def generate_from_ascii(self):
        target = filedialog.askdirectory()
        if not target:
            return

        ascii_to_folders(self.ascii_box.get("0.0", "end"), target)
        messagebox.showinfo("Success", "Folder structure created!")


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app = FolderStudio()
    app.mainloop()
