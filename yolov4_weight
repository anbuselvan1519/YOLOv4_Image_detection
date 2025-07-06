import os
import gdown


url = "https://drive.google.com/file/d/1HuoMhfuTVw99tuTP3gHGXT_Fdqn5MchH/view?usp=sharing"
output_dir = "weights"
output_file = os.path.join(output_dir, "yolov5s.pt")


os.makedirs(output_dir, exist_ok=True)

print("📥 Downloading YOLOv5 weights...")
gdown.download(url, output_file, quiet=False, fuzzy=True)
print(f"✅ Weights downloaded successfully to {output_file}")
