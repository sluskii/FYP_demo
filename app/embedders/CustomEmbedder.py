import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Tuple, Dict, Any, Optional
import logging

from .BaseEmbedder import BaseEmbedder 
# Assuming DEVICE is imported from configs.py along with other constants
from configs import GRID_SIZE, CUSTOM_TEXT_MODEL, CUSTOM_PROJECTION_LAYER, DEVICE 

logger = logging.getLogger(__name__)

class CustomEmbedder(BaseEmbedder):
    def __init__(self, image_folder: str, num_workers: int):
        super().__init__(image_folder, num_workers)
        
        if CUSTOM_TEXT_MODEL is None:
            logger.error("Custom Text Model failed to load. Custom embeddings will not work.")
            
        # 1. Save the target device (mps/cuda/cpu)
        self.device = DEVICE 
        
        # 2. Assign models/layers (already moved to DEVICE in configs.py)
        self.text_model = CUSTOM_TEXT_MODEL
        self.projection_layer = CUSTOM_PROJECTION_LAYER
        self.grid_size = GRID_SIZE

    def _extract_image_features(self, image_path: str) -> Tuple[List[str], List[List[int]], Image.Image]:
        """Extracts text, bounding boxes, and the image object using Tesseract."""
        page = Image.open(image_path)
        if page.mode != "RGB": page = page.convert("RGB")
        
        data = pytesseract.image_to_data(page, output_type=pytesseract.Output.DICT)
        all_words, all_boxes = [], []
        
        # NOTE: Ensure all necessary OCR libraries are installed (like pytesseract and tesseract itself)
        for i in range(len(data['text'])):
            if data['text'][i].strip() != '':
                all_words.append(data['text'][i])
                all_boxes.append([data['left'][i], data['top'][i], 
                                data['left'][i]+data['width'][i], 
                                data['top'][i]+data['height'][i]])
        return all_words, all_boxes, page

    def _get_text_embedding(self, all_words: List[str]) -> np.ndarray:
        """Generates a text embedding from a list of words."""
        # SentenceTransformer handles its own device placement based on where it was loaded (self.text_model)
        # It returns a NumPy array, which must be converted to a tensor and moved to self.device later.
        return self.text_model.encode(" ".join(all_words))

    def _create_distorted_grid(self, bboxes: List[List[int]], original_page_size: Tuple[int, int], grid_size: Tuple[int, int]) -> np.ndarray:
        """Creates a normalized 2D density grid from bounding boxes."""
        grid_width, grid_height = grid_size
        page_width, page_height = original_page_size
        grid = np.zeros((grid_height, grid_width))
        x_scale, y_scale = grid_width / page_width, grid_height / page_height
        for x0, y0, x4, y4 in bboxes:
            grid_x0 = min(max(int(x0 * x_scale), 0), grid_width - 1)
            grid_y0 = min(max(int(y0 * y_scale), 0), grid_height - 1)
            grid_x4 = min(max(int(x4 * x_scale), 0), grid_width - 1)
            grid_y4 = min(max(int(y4 * y_scale), 0), grid_height - 1)
            # Ensure indices are valid (should be covered by min/max but safety check)
            if grid_y0 <= grid_y4 and grid_x0 <= grid_x4:
                 grid[grid_y0 : grid_y4 + 1, grid_x0 : grid_x4 + 1] += 1
        
        max_count = np.max(grid)
        return grid / max_count if max_count > 0 else grid

    def process_image(self, img_filename: str) -> Optional[Dict[str, Any]]:
        """
        Processes a single image using the custom pipeline (implements abstract method).
        """
        if self.text_model is None:
            return None
            
        image_uuid, _ = os.path.splitext(img_filename)
        image_path = os.path.join(self.image_folder, img_filename)
        
        try:
            # 1. Extract features
            all_words, all_boxes, page = self._extract_image_features(image_path)
            if not all_words: return None 

            # 2. Get embeddings
            text_embedding = self._get_text_embedding(all_words)
            distorted_grid = self._create_distorted_grid(all_boxes, page.size, self.grid_size)
            layout_embedding = distorted_grid.flatten()

            # 3. Combine and Project
            # Convert NumPy arrays to PyTorch tensors (still on CPU)
            text_embedding_tensor = torch.from_numpy(text_embedding).float()
            flat_grid_embedding_tensor = torch.from_numpy(layout_embedding).float()
            
            # >>> FIX: Move ALL input tensors to the target device (mps) <<<
            text_embedding_tensor = text_embedding_tensor.to(self.device)
            flat_grid_embedding_tensor = flat_grid_embedding_tensor.to(self.device)
            
            # Concatenate the device-placed tensors
            combined_tensor = torch.cat((text_embedding_tensor, flat_grid_embedding_tensor))
            
            # The projection layer (on self.device) can now accept the input tensor (on self.device)
            with torch.no_grad():
                final_embedding = self.projection_layer(combined_tensor)
            
            return {
                'image_uuid': image_uuid,
                'text_embedding': text_embedding.tolist(), # Return original numpy list for storage
                'layout_embedding': layout_embedding.tolist(), # Return original numpy list for storage
                # Move the final embedding back to CPU and convert to list for saving
                'combined_embedding': final_embedding.cpu().numpy().tolist() 
            }
        except Exception as e:
            logger.error(f"Error custom processing image {img_filename}: {e}", exc_info=True)
            return None