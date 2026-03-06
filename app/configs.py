import torch.nn as nn
# Import LayoutLM classes but its initialization block will be disabled for memory
from transformers import LayoutLMModel, LayoutLMTokenizer 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer
import torch 
import logging
import warnings
import os

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Filter Hugging Face warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

# --- Determine Device and Optimal Dtype for Memory Efficiency ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32  # MPS does not support bfloat16
    logger.info(f"Using Apple MPS device with {DTYPE}.")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    logger.info(f"Using CUDA device with {DTYPE}.")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
    logger.info(f"Using CPU with {DTYPE}. Memory warnings expected for large models.")


# ====================================================================
# --- 1. CONFIGURATION CONSTANTS ---
# ====================================================================

GRID_SIZE = (50, 50)
TEXT_EMBED_DIM = 384
LAYOUT_EMBED_DIM = GRID_SIZE[0] * GRID_SIZE[1]
CUSTOM_INPUT_DIM = TEXT_EMBED_DIM + LAYOUT_EMBED_DIM # 2884

# ====================================================================
# --- 2. MODEL INITIALIZATION (Heavy objects loaded once) ---
# ====================================================================

# Custom Text Embedding Model (Sentence Transformer)
try:
    CUSTOM_TEXT_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # FIX: Explicitly move SentenceTransformer to the determined device
    CUSTOM_TEXT_MODEL.to(DEVICE) 
    logger.info("SentenceTransformer model loaded successfully and moved to device.")
except Exception as e:
    logger.warning(f"SentenceTransformer model loading failed. Custom text embedding may fail. Error: {e}")
    CUSTOM_TEXT_MODEL = None


# Custom Projection Layer (nn.Linear)
CUSTOM_PROJECTION_LAYER = nn.Linear(in_features=CUSTOM_INPUT_DIM, out_features=CUSTOM_INPUT_DIM)
# FIX: Explicitly move custom layer to the determined device
CUSTOM_PROJECTION_LAYER.to(DEVICE)


# LayoutLM Model Initialization - DISABLED FOR MEMORY CONSERVATION
# The model load wa`s stalling/killing the process; setting to None frees up memory.
# LAYOUTLM_MODEL = None
# LAYOUTLM_TOKENIZER = None
# LAYOUTLM_PROCESSOR = None
# logger.warning("LayoutLM initialization is disabled to conserve memory.")

# # LayoutLM classes are imported above, but the initialization block below is commented out:
# try:
#     LAYOUTLM_MODEL = LayoutLMModel.from_pretrained(
#         "microsoft/layoutlm-base-uncased",
#         torch_dtype=DTYPE,
#         device_map="auto"
#     )
#     LAYOUTLM_TOKENIZER = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
#     LAYOUTLM_MODEL.eval()
    
#     LAYOUTLM_PROCESSOR = LAYOUTLM_TOKENIZER 
#     logger.info("LayoutLM model loaded successfully with memory optimization.")
# except Exception as e:
#     logger.warning(f"LayoutLM model loading failed. LayoutLM generation will be disabled. Error: {e}")
#     LAYOUTLM_MODEL, LAYOUTLM_TOKENIZER = None, None 
#     LAYOUTLM_PROCESSOR = None

# LayoutLMv3 Checkpoint
LAYOUTLM3_CHECKPOINT = "microsoft/layoutlmv3-base"

# --- LayoutLMv3 Model Initialization (Re-enabled) ---
try:
    # Use AutoModel/AutoTokenizer/AutoProcessor for best compatibility
    # Attempt to load with device_map='auto' for memory optimization, but fall
    # back to a single-device load if the model class doesn't support it
    try:
        LAYOUTLM3_MODEL = AutoModel.from_pretrained(
            LAYOUTLM3_CHECKPOINT,
            torch_dtype=DTYPE,
            device_map="auto"
        )
        logger.info("LayoutLMv3 loaded with device_map='auto'.")
    except ValueError as ve:
        # Some model classes (e.g., LayoutLMv3) don't implement support for
        # `device_map='auto'` and will raise a ValueError. Fall back to a
        # standard load and move the model to the determined device.
        logger.warning("device_map='auto' unsupported for LayoutLMv3, falling back to single-device load. Error: %s", ve)
        LAYOUTLM3_MODEL = AutoModel.from_pretrained(
            LAYOUTLM3_CHECKPOINT,
            torch_dtype=DTYPE
        )
        try:
            LAYOUTLM3_MODEL.to(DEVICE)
        except Exception:
            # Some wrappers may not implement `.to`; ignore if moving fails
            pass

    LAYOUTLM3_TOKENIZER = AutoTokenizer.from_pretrained(LAYOUTLM3_CHECKPOINT)

    # LayoutLMv3 also uses a Processor which handles image and text inputs
    # Set apply_ocr=False since Tesseract is handling OCR in the embedder class
    LAYOUTLM3_PROCESSOR = AutoProcessor.from_pretrained(LAYOUTLM3_CHECKPOINT, apply_ocr=False)

    LAYOUTLM3_MODEL.eval()

    logger.info(f"LayoutLMv3 model loaded successfully from {LAYOUTLM3_CHECKPOINT} with memory optimization.")
except Exception as e:
    logger.error(f"LayoutLMv3 model loading failed. LayoutLM generation will be disabled. Error: {e}", exc_info=True)
    LAYOUTLM3_MODEL, LAYOUTLM3_TOKENIZER = None, None
    LAYOUTLM3_PROCESSOR = None
    logger.warning("LayoutLMv3 initialization failed and remains disabled.")

# --- Local Cache Path ---
LOCAL_DOCLLM_PATH = os.path.join(os.getcwd(), "local_docllm_cache")
DOCLLM_MODEL_ID = "minlik/docllm-yi-6b" 

try:
    if os.path.isdir(LOCAL_DOCLLM_PATH):
        logger.info(f"Loading DocLLM model from local cache: {LOCAL_DOCLLM_PATH}")
        # Load from the local directory
        model_source = LOCAL_DOCLLM_PATH
    else:
        logger.info(f"Local cache not found. Will download DocLLM model from Hugging Face: {DOCLLM_MODEL_ID}")
        # Set source to the remote ID for download
        model_source = DOCLLM_MODEL_ID

    # 1. Load Tokenizer
    DOCLLM_TOKENIZER = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True
    )
    
    # 2. Load Model
    DOCLLM_MODEL = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=DTYPE,         
        device_map="auto",         
        trust_remote_code=True
    )

    # 3. Save Model Locally if it was downloaded remotely
    if model_source == DOCLLM_MODEL_ID:
        logger.info(f"Saving DocLLM model to local cache: {LOCAL_DOCLLM_PATH}")
        DOCLLM_MODEL.save_pretrained(LOCAL_DOCLLM_PATH)
        DOCLLM_TOKENIZER.save_pretrained(LOCAL_DOCLLM_PATH)

    DOCLLM_MODEL.eval()
    logger.info(f"DocLLM model loaded successfully from {model_source} with dtype {DTYPE}.")
    
except Exception as e:
    logger.error(f"DocLLM model loading failed. DocLLM features will be disabled. Error: {e}")
    DOCLLM_MODEL, DOCLLM_TOKENIZER = None, None