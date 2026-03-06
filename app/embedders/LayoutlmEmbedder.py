import os
import torch
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Tuple, Dict, Any, Optional
import logging
from transformers import LayoutLMv3Processor

# Import the shared constants that hold the loaded LayoutLMv3 model/tokenizer
from configs import LAYOUTLM3_MODEL, LAYOUTLM3_TOKENIZER, DEVICE 
from .BaseEmbedder import BaseEmbedder 

logger = logging.getLogger(__name__)

# --- CORE EMBEDDER CLASS (LayoutLMv3) ---

class Layoutlmv3Embedder(BaseEmbedder):
    """
    Embedder using the LayoutLMv3 model, loaded from the global config constants.
    """
    def __init__(self, image_folder: str, num_workers: int):
        super().__init__(image_folder, num_workers)
        
        if LAYOUTLM3_MODEL is None or LAYOUTLM3_TOKENIZER is None:
            logger.error("LayoutLMv3 model or tokenizer failed to load in configs. Embeddings will not work.")
            
        self.model = LAYOUTLM3_MODEL
        self.tokenizer = LAYOUTLM3_TOKENIZER
        
        # Initialize LayoutLMv3Processor for better image/OCR handling
        try:
            self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
        except Exception as e:
            logger.warning(f"Could not load LayoutLMv3Processor: {e}. Falling back to manual OCR.")
            self.processor = None
            
        self.device = DEVICE

    def _normalize_box(self, box: List[int], width: int, height: int) -> List[int]:
        """Normalizes bounding box coordinates to a 0-1000 scale."""
        scale = 1000
        return [
            int(scale * (box[0] / width)),
            int(scale * (box[1] / height)),
            int(scale * (box[2] / width)),
            int(scale * (box[3] / height)),
        ]

    def _preprocess_image_for_layoutlmv3(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image to generate pixel_values for LayoutLMv3."""
        try:
            if self.processor is not None:
                # Use LayoutLMv3Processor for optimal image preprocessing
                encoding = self.processor(image, return_tensors="pt")
                return encoding['pixel_values']
            else:
                # Fallback: Manual image preprocessing
                # Resize to LayoutLMv3 expected size (224x224)
                image_resized = image.resize((224, 224))
                
                # Convert to tensor and normalize (standard ImageNet normalization)
                image_array = np.array(image_resized).astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_normalized = (image_array - mean) / std
                
                # Convert to tensor: (H, W, C) -> (1, C, H, W)
                pixel_values = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0)
                return pixel_values
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return dummy pixel values as fallback
            return torch.zeros(1, 3, 224, 224)

    def _extract_image_features(self, image_path: str) -> Tuple[List[str], List[List[int]], Image.Image]:
        """Extracts text, *normalized* bounding boxes, and the image object using Tesseract."""
        page = Image.open(image_path)
        if page.mode != "RGB": page = page.convert("RGB")
        width, height = page.size
        
        data = pytesseract.image_to_data(page, output_type=pytesseract.Output.DICT)
        all_words, normalized_boxes = [], []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text:
                all_words.append(text)
                # Raw box
                raw_box = [data['left'][i], data['top'][i], 
                           data['left'][i]+data['width'][i], 
                           data['top'][i]+data['height'][i]]
                # Normalize and store
                normalized_boxes.append(self._normalize_box(raw_box, width, height))

        return all_words, normalized_boxes, page

    def _generate_layoutlmv3_embedding(self, words: List[str], boxes: List[List[int]], pixel_values: torch.Tensor, max_seq_length: int = 512) -> List[float]:
        """
        Generates a single LayoutLMv3 embedding for a document using the [CLS] token representation.
        FIXED: Now includes pixel_values for visual understanding.
        """
        if self.model is None or self.tokenizer is None:
            return []

        # Use the tokenizer's built-in methods for text encoding and box alignment
        encoding = self.tokenizer(
            words, 
            boxes=boxes, 
            padding="max_length", 
            truncation=True, 
            max_length=max_seq_length,
            return_tensors="pt"
        )
        
        # --- Create Tensors and Move to Device ---
        input_ids_t = encoding['input_ids'].to(self.device)
        attention_mask_t = encoding['attention_mask'].to(self.device)
        token_boxes_t = encoding['bbox'].to(self.device)
        pixel_values_t = pixel_values.to(self.device)
        
        # Inference - FIXED: Include pixel_values for visual features
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_t,
                bbox=token_boxes_t,
                attention_mask=attention_mask_t,
                pixel_values=pixel_values_t  # CRITICAL FIX: Add visual features!
            )
        
        # Return [CLS] token embedding (index 0)
        # FIX: Convert BFloat16 to Float32 before moving to CPU and converting to numpy
        cls_embedding = outputs.last_hidden_state[0, 0, :]
        return cls_embedding.to(torch.float32).cpu().numpy().tolist()

    def process_image(self, img_filename: str) -> Optional[Dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            return None
            
        image_uuid, _ = os.path.splitext(img_filename)
        image_path = os.path.join(self.image_folder, img_filename)

        try:
            # 1. Extract features using private method
            all_words, all_boxes, image = self._extract_image_features(image_path)
            
            if not all_words:
                logger.warning(f"No text extracted for image: {img_filename}")
                return None
            
            # 2. FIXED: Preprocess image for visual features
            pixel_values = self._preprocess_image_for_layoutlmv3(image)
            
            # 3. Generate LayoutLMv3 embedding with visual features
            document_embedding = self._generate_layoutlmv3_embedding(all_words, all_boxes, pixel_values)
            
            return {
                'image_uuid': image_uuid,
                'layoutlmv3_embedding': document_embedding
            }
        except Exception as e:
            logger.error(f"Error LayoutLMv3 processing image {img_filename}: {e}", exc_info=True)
            return None