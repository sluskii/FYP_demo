import os
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Set up logging for the base class
logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    def __init__(self, image_folder: str, num_workers: int):
        self.image_folder = image_folder
        self.num_workers = num_workers
        self.image_files = [
            img for img in os.listdir(image_folder) 
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not self.image_files:
            logger.warning(f"No images found in folder: {image_folder}")

    @abstractmethod
    def process_image(self, img_filename: str) -> Optional[Dict[str, Any]]:
        """Processes a single image and returns a dictionary of results."""
        pass

    def generate_embeddings(self, desc: str = "Generating Embeddings") -> pd.DataFrame:
        """
        Runs the multi-threaded pipeline using the concrete process_image method 
        implemented by the child classes.
        """
        results: List[Dict[str, Any]] = []
        
        if not self.image_files:
            logger.info("No image files to process.")
            return pd.DataFrame()

        # The threading logic is centralized here
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_img = {
                # Submit the abstract method (which the child class must implement)
                executor.submit(self.process_image, img_filename): img_filename
                for img_filename in self.image_files
            }

            for future in tqdm(as_completed(future_to_img), total=len(self.image_files), desc=desc):
                result = future.result()
                if result is not None:
                    results.append(result)

        return pd.DataFrame(results)