import os
import cv2
import torch
import numpy as np
import pandas as pd
from tkinter import Tk, Canvas, Button, Label
from PIL import Image, ImageTk
from segment_anything import sam_model_registry, SamPredictor

# Config
IMAGE_DIR = "water_dataset/images"
MASK_DIR = "water_dataset/masks"
METADATA_PATH = "water_dataset/metadata.csv"
CHECKPOINT_PATH = "segment-anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

# Setup
os.makedirs(MASK_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device)
predictor = SamPredictor(sam)

# Session memory
metadata_rows = []
pending_masks = []

# Tkinter App
class MaskApp:
    def __init__(self, master):
        self.master = master
        self.master.title("🚀 SAM Mask Creator (Optimized)")
        self.image_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png')))
        self.image_index = 0
        self.clicks = []
        self.masks = []

        self.canvas = Canvas(master, width=800, height=600)
        self.canvas.pack()
        self.label = Label(master, text="")
        self.label.pack()

        Button(master, text="Add Mask", command=self.add_mask).pack(side="left")
        Button(master, text="Retry", command=self.retry).pack(side="left")
        Button(master, text="Next Image", command=self.next_image).pack(side="left")
        Button(master, text="Skip Image", command=self.skip_image).pack(side="left")
        Button(master, text="Save All & Exit", command=self.save_all_and_exit).pack(side="left")

        self.canvas.bind("<Button-1>", self.on_click)
        self.load_image()

    def load_image(self):
        self.clicks = []
        self.masks = []
        if self.image_index >= len(self.image_files):
            self.label.config(text="✅ All images done. Click 'Save All & Exit'")
            return

        filename = self.image_files[self.image_index]
        self.current_image_path = os.path.join(IMAGE_DIR, filename)
        self.original = cv2.imread(self.current_image_path)
        self.rgb_image = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        predictor.set_image(self.rgb_image)
        self.show_image(self.rgb_image)
        self.label.config(text=f"🖼️ Image: {filename}")

    def show_image(self, img_array):
        img_pil = Image.fromarray(img_array)
        img_pil = img_pil.resize((800, 600))
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def on_click(self, event):
        x = int(event.x * self.rgb_image.shape[1] / 800)
        y = int(event.y * self.rgb_image.shape[0] / 600)
        print(f"🖱️ Clicked: ({x}, {y})")
        self.clicks.append((x, y))
        self.generate_mask(x, y)

    def generate_mask(self, x, y):
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False  # Just one mask = faster
        )
        mask = masks[0]
        score = scores[0]
        self.masks.append((mask, x, y, float(score)))

        # Show combined overlay
        overlay = self.rgb_image.copy()
        combined = np.zeros_like(mask, dtype=bool)
        for m, *_ in self.masks:
            combined = np.logical_or(combined, m)
        overlay[combined] = [0, 255, 0]
        self.show_image(overlay)

    def add_mask(self):
        print("➕ Added mask. Click again or go to next image.")

    def retry(self):
        print("🔁 Retrying image.")
        self.load_image()

    def next_image(self):
        if not self.masks:
            print("⚠️ No masks, skipping to next.")
            self.image_index += 1
            self.load_image()
            return

        # Combine all masks into one
        combined_mask = np.zeros_like(self.masks[0][0], dtype=bool)
        click_info = []
        for mask, x, y, score in self.masks:
            combined_mask = np.logical_or(combined_mask, mask)
            click_info.append((x, y, score))

        image_name = self.image_files[self.image_index]
        mask_name = f"{os.path.splitext(image_name)[0]}_mask.png"
        mask_path = os.path.join(MASK_DIR, mask_name)

        # Save mask later
        pending_masks.append((mask_path, combined_mask))

        area = int(np.sum(combined_mask))
        bbox = cv2.boundingRect(combined_mask.astype(np.uint8))
        avg_score = np.mean([s for _, _, s in click_info])

        metadata_rows.append({
            "image": image_name,
            "mask": mask_name,
            "click_x": [x for x, _, _ in click_info],
            "click_y": [y for _, y, _ in click_info],
            "area": area,
            "bbox": bbox,
            "score": avg_score
        })

        print(f"✅ Queued mask for {image_name}")
        self.image_index += 1
        self.load_image()

    def skip_image(self):
        print("⏭️ Skipped image.")
        self.image_index += 1
        self.load_image()

    def save_all_and_exit(self):
        # Save all masks
        for path, mask in pending_masks:
            cv2.imwrite(path, (mask.astype(np.uint8) * 255), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"🧾 Saved {len(pending_masks)} masks.")

        # Save metadata
        df = pd.DataFrame(metadata_rows)
        df.to_csv(METADATA_PATH, index=False)
        print(f"📄 Metadata saved to {METADATA_PATH}")

        self.master.quit()


# Run the App
root = Tk()
app = MaskApp(root)
root.mainloop()
