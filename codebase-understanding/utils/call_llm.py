from google import genai
import os
import logging
import json
from datetime import datetime
import time
from threading import Lock
import dotenv

dotenv.load_dotenv()


# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"

# LLM Provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # "gemini" or "openai"

# Rate limiting configuration (tokens per minute)
if LLM_PROVIDER == "openai":
    TPM_LIMIT = int(os.getenv("OPENAI_TPM_LIMIT", "200000"))  # OpenAI has higher limits
else:
    TPM_LIMIT = int(os.getenv("GEMINI_TPM_LIMIT", "200000"))

_tokens = TPM_LIMIT
_last_check = time.monotonic()
_bucket_lock = Lock()

# Try to import required libraries
_tokenizer = None
HAS_OPENAI = False
HAS_TOKENIZER = False

try:
    import tiktoken
    if LLM_PROVIDER == "openai":
        _tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")  # Use model-specific encoding
    else:
        _tokenizer = tiktoken.get_encoding("cl100k_base")  # Good approximation for Gemini
    HAS_TOKENIZER = True
    print(f"Using tiktoken for accurate token counting ({LLM_PROVIDER})")
except ImportError:
    HAS_TOKENIZER = False
    print("tiktoken not found, using estimation (install with: pip install tiktoken)")

if LLM_PROVIDER == "openai":
    try:
        from openai import OpenAI
        HAS_OPENAI = True
        print("OpenAI client available")
    except ImportError:
        print("OpenAI client not found (install with: pip install openai)")
        HAS_OPENAI = False


def _refill_tokens():
    """Refill tokens in the bucket based on elapsed time."""
    global _tokens, _last_check
    now = time.monotonic()
    elapsed = now - _last_check
    if elapsed <= 0:
        return
    rate_per_sec = TPM_LIMIT / 60
    _tokens = min(TPM_LIMIT, _tokens + elapsed * rate_per_sec)
    _last_check = now


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken if available, otherwise estimate."""
    if not text:
        return 0
    
    if HAS_TOKENIZER and _tokenizer:
        try:
            return len(_tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Tokenizer failed, falling back to estimation: {e}")
    
    # Fallback estimation - better for code with whitespace/punctuation
    return max(1, len(text) // 6)


def _acquire_tokens(tokens_needed: float) -> None:
    """Block until the requested number of tokens is available."""
    global _tokens
    
    # Cap at 80% of bucket to prevent infinite waits
    max_allowed = TPM_LIMIT * 0.8
    if tokens_needed > max_allowed:
        logger.warning(f"Request needs {tokens_needed} tokens, capping at {max_allowed}")
        tokens_needed = max_allowed
    
    rate_per_sec = TPM_LIMIT / 60
    while True:
        with _bucket_lock:
            _refill_tokens()
            if tokens_needed <= _tokens:
                _tokens -= tokens_needed
                return
            missing = tokens_needed - _tokens
        
        wait_time = missing / rate_per_sec
        logger.info(f"Rate limiting: waiting {wait_time:.1f}s for {missing:.0f} tokens")
        time.sleep(wait_time)


def _call_openai(prompt: str) -> str:
    """Call OpenAI API."""
    if not HAS_OPENAI:
        raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000  # Adjust as needed
        )
        
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("Empty response from OpenAI API")
            
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise ValueError(f"Failed to get response from OpenAI: {e}")


def _call_gemini(prompt: str) -> str:
    """Call Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    try:
        # Option 1: Vertex AI (uncomment if using)
        # client = genai.Client(
        #     vertexai=True,
        #     project=os.getenv("GEMINI_PROJECT_ID", "your-project-id"),
        #     location=os.getenv("GEMINI_LOCATION", "us-central1")
        # )

        # Option 2: AI Studio API key
        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        
        response = client.models.generate_content(model=model, contents=[prompt])
        
        if not response or not response.text or response.text.strip() == "":
            raise ValueError("Empty response from Gemini API")
            
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise ValueError(f"Failed to get response from Gemini: {e}")


def call_llm(prompt: str, use_cache: bool = True) -> str:
    """Call LLM (OpenAI or Gemini) with optional caching and token-per-minute limiting."""
    #print(f"DEBUG: LLM_PROVIDER='{LLM_PROVIDER}', TPM_LIMIT={TPM_LIMIT}")
    if not prompt or not prompt.strip():
        raise ValueError("Empty prompt provided")
    
    # Log the prompt
    logger.info(f"PROMPT ({LLM_PROVIDER}): {prompt}")
    
    # Pre-check token count
    prompt_tokens = _count_tokens(prompt)
    logger.info(f"Prompt token count: {prompt_tokens}")
    
    # Check if prompt is too large
    if prompt_tokens > TPM_LIMIT:
        raise ValueError(f"Prompt too large: {prompt_tokens} tokens exceeds TPM limit of {TPM_LIMIT}")
    
    # Check cache if enabled
    cache = {}
    if use_cache:
        # Load cache from disk
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Use provider-specific cache key to avoid conflicts
        cache_key = f"{LLM_PROVIDER}:{prompt}"
        
        # Return from cache if exists (no rate limiting needed)
        if cache_key in cache:
            logger.info(f"Response found in cache ({LLM_PROVIDER})")
            logger.info(f"RESPONSE: {cache[cache_key]}")
            return cache[cache_key]

    # Only throttle if we're actually making an API call
    _acquire_tokens(prompt_tokens)

    # Call the appropriate LLM
    if LLM_PROVIDER == "openai":
        response_text = _call_openai(prompt)
    elif LLM_PROVIDER == "gemini":
        response_text = _call_gemini(prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}. Use 'openai' or 'gemini'")

    # Log the response
    logger.info(f"RESPONSE ({LLM_PROVIDER}): {response_text}")

    # Update cache if enabled
    if use_cache:
        # Load cache again to avoid overwrites from concurrent processes
        try:
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to reload cache for update: {e}")

        # Add to cache with provider-specific key
        cache_key = f"{LLM_PROVIDER}:{prompt}"
        cache[cache_key] = response_text
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    return response_text


if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    print(f"Testing {LLM_PROVIDER.upper()} provider...")
    
    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")