import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class SpineDataset(Dataset):
    def __init__(self, img_paths, mask_paths, atlas_paths, image_size=512):
        """
        Args:
            img_paths (list): List of file paths to the target CT images (2D).
            mask_paths (list): List of file paths to the ground truth masks (multi-class).
            atlas_paths (list): List of lists. Each element is a list of file paths to the atlas masks 
                                corresponding to the target image at that index. 
                                Length of outer list must match img_paths.
                                Length of inner list must match num_atlases expected by the model.
            image_size (int): Target size for resizing (default 512).
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.atlas_paths = atlas_paths
        self.image_size = image_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # 1. Load Image
        img_path = self.img_paths[index]
        # Open image and convert to grayscale (L) just in case, though CT is already 1-channel.
        # If input is already RGB, convert to L to standardize, then replicate.
        img = Image.open(img_path).convert('L') 

        # 2. Load Mask (Ground Truth)
        mask_path = self.mask_paths[index]
        # Open mask. Do NOT convert to 'L' blindly if it's already integer class map.
        # Assuming mask is an image where pixel values represent classes (0, 1, 2...).
        mask = Image.open(mask_path)

        # 3. Load Atlases (Prompts)
        atlas_entry = self.atlas_paths[index]
        atlas_masks = []
        # atlas_entry should be a list of paths to the N atlas masks for this image
        for ap in atlas_entry:
            am = Image.open(ap)
            # Resize atlas mask using NEAREST to preserve class values if they are masks
            # If they are probability maps, use BILINEAR. Assuming masks based on variable name 'atlas_mask'.
            am = am.resize((self.image_size, self.image_size), resample=Image.NEAREST)
            am_np = np.array(am)
            atlas_masks.append(torch.from_numpy(am_np).float())
        
        # Stack atlases along channel dimension: (Num_Atlases, H, W)
        atlas_tensor = torch.stack(atlas_masks, dim=0)

        # 4. Preprocessing & Resizing
        # Resize Image (Bilinear for continuous values)
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        
        # Resize GT Mask (Nearest for categorical values)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        # 5. Convert to Tensor
        # Image: (H, W) -> (1, H, W) -> (3, H, W)
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0) 
        img_tensor = img_tensor.repeat(3, 1, 1) # Replicate to 3 channels for ViT encoder
        img_tensor = img_tensor / 255.0 # Normalize to [0, 1]

        # Mask: (H, W)
        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy(mask_np).long() # Integer for classes

        return img_tensor, mask_tensor, atlas_tensor