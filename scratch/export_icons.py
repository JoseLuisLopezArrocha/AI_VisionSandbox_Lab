from PIL import Image, ImageDraw, ImageFont
import os

ICON_DIR = "assets/icons"
os.makedirs(ICON_DIR, exist_ok=True)

# Iconos a exportar (MDL2 Assets)
ICONS = {
    "source": "\uE724",    # Plug
    "favs": "\uE734",      # Star
    "alerts": "\uE7ED",    # Bell
    "refresh": "\uE72C",   # Refresh
    "night": "\uE708",     # Moon
    "day": "\uE706",       # Sun
    "settings": "\uE713",  # Gear
    "info": "\uE946",      # Info
    "play": "\uE768",      # Play
    "pause": "\uE769",     # Pause
    "back": "\uE892",      # Back
    "forward": "\uE893",   # Forward
    "models": "\uE950",    # Processor/AI
}

def export_icon(name, char, size=32, color=(200, 200, 200)):
    # Intentar cargar Segoe MDL2 Assets
    try:
        # En Windows suele estar en C:\Windows\Fonts\segmdl2.ttf
        font_path = "C:\\Windows\\Fonts\\segmdl2.ttf"
        font = ImageFont.truetype(font_path, int(size * 0.8))
    except:
        # Fallback a un font por defecto si no estamos en Windows
        font = ImageFont.load_default()

    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Centrar el texto
    # Usamos textbbox para centrar
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    draw.text(((size - w) // 2, (size - h) // 2 - 2), char, font=font, fill=color)
    
    img.save(os.path.join(ICON_DIR, f"{name}.png"))

for name, char in ICONS.items():
    print(f"Exporting {name}...")
    export_icon(name, char)
    # Exportar también en negro para tema claro
    export_icon(name + "_dark", char, color=(30, 30, 30))

print("Icons exported to assets/icons/")
