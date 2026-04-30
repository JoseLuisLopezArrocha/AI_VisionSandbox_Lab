import os
import shutil

def test_find_first_unlabeled():
    # Setup mock dataset
    test_ds = "test_dataset"
    img_dir = os.path.join(test_ds, "images", "train")
    lab_dir = os.path.join(test_ds, "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    
    image_files = []
    for i in range(5):
        path = os.path.join(img_dir, f"img_{i}.jpg")
        with open(path, "w") as f: f.write("fake image")
        image_files.append(path)
    
    # Label first 2
    for i in range(2):
        with open(os.path.join(lab_dir, f"img_{i}.txt"), "w") as f: f.write("0 0.5 0.5 0.1 0.1")
        
    print(f"Total imágenes: {len(image_files)}")
    print("Etiquetadas: img_0, img_1")
    
    # Mock find logic
    def find_first(files, ds_dir):
        for idx, p in enumerate(files):
            base = os.path.splitext(os.path.basename(p))[0]
            label_path = os.path.join(ds_dir, "labels", "train", f"{base}.txt")
            if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
                return idx
        return 0

    first = find_first(image_files, test_ds)
    print(f"Primera no etiquetada detectada en índice: {first} (Esperado: 2)")
    
    # Cleanup
    shutil.rmtree(test_ds)
    
    if first == 2:
        print("✅ Verificación lógica EXITOSA")
    else:
        print("❌ Verificación lógica FALLIDA")

if __name__ == "__main__":
    test_find_first_unlabeled()
