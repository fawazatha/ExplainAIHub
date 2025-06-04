# ***** IMPORT FRAMEWORK *****
from langchain.chat_models import init_chat_model

# ***** IMPORT TYPE ANOTATION *****
from typing import Any

# ***** IMPORT CUSTOM FUNCTION *****
from config.settings import settings

def load_llm_model(model_llm: str, 
                    load_type: str = 'api',
                    model_type: str = 'gemini') -> Any: 
    """
    Load a large language model (LLM) instance either via API or Hugging Face.

    Args:
        model_llm (str): Identifier or key for the LLM (e.g., API key or model name).
        load_type (str): Method to load the model, either 'api' or 'huggingface'. Defaults to 'api'.
        model_type (str): Specific LLM family, such as 'gemini' or 'gpt'. Defaults to 'gemini'.

    Returns:
        Any: An initialized LLM object ready for inference.
    """
    # ***** Determine loading via API or Hugging Face based on load_type *****
    if load_type.lower() == 'api':
        # ***** Load from API (e.g., Gemini via Google GenAI) *****
        if model_type.lower() == 'gemini':
            try:
                # ***** Use provided model_llm or fall back to environment variable *****
                key = model_llm or settings.GOOGLE_API_KEY
                if not key:
                    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

                # ***** Initialize the Gemini chat model with specific parameters *****
                model = init_chat_model(
                    "google_genai:gemini-2.0-flash",
                    temperature=0.9,
                    max_tokens=8000
                )
                return model
            except Exception as error:
                # ***** Propagate error if Gemini loading fails *****
                raise ValueError("Failed to load Gemini model. Check your API key or model name.") from error

        elif model_type.lower() == 'gpt':
            # ***** Placeholder for future GPT API loading logic *****
            pass
        else:
            # ***** Unsupported model_type error *****
            raise ValueError(f"Unsupported model_type: {model_type}")

    elif load_type.lower() == 'huggingface':
        # ***** Placeholder for Hugging Face loading logic (to be implemented) *****
        pass
    
    # If neither branch is taken, return None or raise as needed
    return None
        
        