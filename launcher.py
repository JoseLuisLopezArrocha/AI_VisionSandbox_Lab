"""Punto de entrada alternativo. El splash screen está integrado en main.py."""
from main import VisionApp

if __name__ == "__main__":
    app = VisionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
