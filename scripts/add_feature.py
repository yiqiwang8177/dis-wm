import torch
import h5py
import hdf5plugin # important to load pixels
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from tqdm import tqdm
import timm 
num_workers = 6
batch_size = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0,)
model.to(DEVICE)
model.eval()
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

def process_one(image_array):
    img = Image.fromarray(image_array)
    return transforms(img)

def get_features(image_arrays):
    # features_list = []
    # for image_array in image_arrays:

    #     test_pixels = Image.fromarray(image_array)
    #     test_pixels = transforms(test_pixels)
    #     features_list.append(test_pixels)
    # --- Parallel preprocessing (ORDER PRESERVED) ---
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        processed = list(executor.map(process_one, image_arrays))
    features = torch.stack(processed).to(DEVICE)  # Add batch dimension and move to device

    with torch.no_grad():
        features = model.forward_features(features)
    return features.cpu().numpy()  # Move features back to CPU and convert to numpy array


if __name__ == "__main__":
    
    hdf5_file = '/home/atkeonlab-3/.stable_worldmodel/tworoom.h5'
    
    hdf5_file_out = '/home/atkeonlab-3/.stable_worldmodel/tworoom_with_features.h5'
    with h5py.File(hdf5_file, "r") as f_in, h5py.File(hdf5_file_out, "w") as f_out:

        pixels = f_in["pixels"]
        T = pixels.shape[0]

        # --- Copy everything EXCEPT pixels ---
        for key in f_in.keys():
            if key != "pixels":
                f_in.copy(key, f_out)

        # --- Determine feature shape ---
        # Run one small batch to infer shape
        sample_feat = get_features(pixels[0:1])
        tokens, d = sample_feat.shape[1], sample_feat.shape[2]

        # --- Create output dataset ---
        features_ds = f_out.create_dataset(
            "feature",
            shape=(T, tokens, d),
            dtype="float32",
            chunks=(batch_size, tokens, d),  # important for performance
            compression="gzip",              # safe compression
            compression_opts=4,
        )

        # --- Main loop ---
        for i in tqdm( range(0, T, batch_size) ):
            batch = pixels[i:i + batch_size]           # (B, H, W, 3)
            feats = get_features(batch)            # (B, tokens, d)
            features_ds[i:i + batch_size] = feats      # write directly

            
    print("Done!")