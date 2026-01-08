import os
import time
import logging
import threading
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QueryLogic")

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    logger.error("❌ CRITICAL: API Keys missing in .env")

genai.configure(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL = "models/text-embedding-004"

# --- 2. GLOBAL SEMAPHORE (TRAFFIC COP) ---
api_limiter = threading.Semaphore(5)

# --- 3. THREAD-LOCAL STORAGE (THE FIX) ---
# This creates a private "pocket" for each thread to store its connection
thread_local_storage = threading.local()

def get_pinecone_index():
    """
    Retrieves the Pinecone index for the current thread.
    If it doesn't exist yet for this thread, it creates it once.
    """
    # Check if THIS thread already has an index connection
    if not hasattr(thread_local_storage, "index"):
        # logger.info("🔌 Opening new Pinecone connection for this thread...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        thread_local_storage.index = pc.Index(PINECONE_INDEX_NAME)
    
    # Return the existing, open connection
    return thread_local_storage.index

# --- PART A: BATCH EMBEDDINGS (With Retries) ---

@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
def generate_embeddings_safe(batch):
    """
    Wraps the API call in a retry loop. 
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=batch,
        task_type="retrieval_document",
        title="Educational Notes"
    )
    return result

def get_embeddings_batch(text_chunks):
    embeddings = []
    BATCH_SIZE = 50 
    
    for i in range(0, len(text_chunks), BATCH_SIZE):
        batch = text_chunks[i:i + BATCH_SIZE]
        
        # Filter empty strings
        valid_batch = [text for text in batch if text.strip()]
        if not valid_batch:
            continue
            
        try:
            # CALL THE RETRY-WRAPPED FUNCTION
            with api_limiter:
                result = generate_embeddings_safe(valid_batch)
            
            if 'embedding' in result:
                embeddings.extend(result['embedding'])
            else:
                logger.warning(f"⚠️ Warning: No embeddings returned for batch {i}")
            
            # Static sleep to be nice to the API
            time.sleep(0.5) 
            
        except Exception as e:
            logger.error(f"❌ Batch Embedding Failed permanently (Batch {i}): {e}")
            
    return embeddings

# --- PART B: UPLOAD LOGIC (Thread-Safe) ---

def add_to_vector_db(chunks, user_id, subject_list, filename):
    """
    Uploads vectors to Pinecone. Thread-safe and robust.
    """
    clean_user_id = user_id.strip()
    normalized_subjects = [s.strip().lower() for s in subject_list]

    logger.info(f"--> 🛠️ Processing {len(chunks)} chunks for user: {clean_user_id}...")
    
    # 1. Get Embeddings
    vectors = get_embeddings_batch(chunks)
    
    if not vectors: 
        logger.error("❌ No vectors generated. Skipping upload.")
        return

    # 2. Get Thread-Local Connection
    index = get_pinecone_index()

    # 3. Prepare Records
    records = []
    limit = min(len(chunks), len(vectors))
    
    for i in range(limit):
        chunk = chunks[i]
        vector = vectors[i]
        
        if vector is None: continue 

        vector_id = f"{clean_user_id}_{filename}_{i}"
        
        metadata = {
            "text": chunk,
            "user_id": clean_user_id,
            "subject": normalized_subjects[0] if normalized_subjects else "general",
            "filename": filename,
            "type": "personal"
        }
        
        records.append({"id": vector_id, "values": vector, "metadata": metadata})

    # 4. Upsert in Batches with Retry Logic
    batch_size = 100
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def upsert_safe(batch_records):
        index.upsert(vectors=batch_records)

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            upsert_safe(batch)
            logger.info(f"   -> ✅ Uploaded batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"   -> ❌ Pinecone Upsert Error after retries: {e}")

# --- PART C: RETRIEVAL LOGIC ---

def ask_pinecone(query, user_id_filter=None, subject_filter=None, top_k=5, threshold=0.45):
    """
    Retrieves relevant chunks. 
    """
    # 1. Get Thread-Local Connection (Fast!)
    index = get_pinecone_index()

    # 2. Generate Query Embedding
    try:
        with api_limiter:
            query_embedding = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query"
            )['embedding']
    except Exception as e:
        logger.error(f"❌ Query Embedding Failed: {e}")
        return []

    # 3. Build Filter
    filter_dict = {}
    if user_id_filter != "NO_FILTER":
        if user_id_filter:
            filter_dict['user_id'] = {"$eq": user_id_filter}
        if subject_filter:
            filter_dict['subject'] = {"$eq": subject_filter.strip().lower()}

    # 4. Query with Retry
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=2))
    def query_safe():
        return index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict 
        )

    try:
        results = query_safe()
        
        matches = []
        for m in results['matches']:
            if m['score'] > threshold:
                matches.append({
                    'text': m['metadata'].get('text', ''),
                    'metadata': m['metadata'],
                    'score': m['score']
                })
        
        return matches

    except Exception as e:
        logger.error(f"❌ Pinecone Query Failed: {e}")
        return []
# import os
# import time
# import asyncio
# import google.generativeai as genai
# from pinecone import Pinecone
# from dotenv import load_dotenv

# # 1. CONFIGURATION
# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") # "edu-tutor-index"

# if not GEMINI_API_KEY or not PINECONE_API_KEY:
#     print("❌ CRITICAL: API Keys missing in .env")

# genai.configure(api_key=GEMINI_API_KEY)

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(PINECONE_INDEX_NAME)

# EMBEDDING_MODEL = "models/text-embedding-004"

# # --- PART A: ASYNC EMBEDDINGS (Your Preference) ---

# async def get_query_embedding_async(text: str):
#     """
#     Generates embedding asynchronously to keep the server fast.
#     """
#     try:
#         # We run the blocking genai call in a thread
#         result = await asyncio.to_thread(
#             genai.embed_content,
#             model=EMBEDDING_MODEL,
#             content=text,
#             task_type="retrieval_query"
#         )
#         return result['embedding']
#     except Exception as e:
#         print(f"❌ Embedding Error: {e}")
#         return None

# def get_embeddings_batch(text_chunks):
#     """
#     Used during Upload: Batches chunks to avoid API limits (Max 100 per call).
#     """
#     embeddings = []
#     BATCH_SIZE = 50
    
#     for i in range(0, len(text_chunks), BATCH_SIZE):
#         batch = text_chunks[i:i + BATCH_SIZE]
#         try:
#             result = genai.embed_content(
#                 model=EMBEDDING_MODEL,
#                 content=batch,
#                 task_type="retrieval_document",
#                 title="Educational Notes"
#             )
#             embeddings.extend(result['embedding'])
#             time.sleep(0.5) # Rate limit safety
#         except Exception as e:
#             print(f"❌ Batch Embedding Error: {e}")
#             embeddings.extend([None] * len(batch))
            
#     return embeddings

# # --- PART B: UPLOAD LOGIC (Dynamic Layer) ---

# # query_logic.py

# # CHANGE 1: Remove 'async'
# def add_to_vector_db(chunks, user_id, subject_list, filename):
#     """
#     Synchronous version compatible with both Threads and Asyncio.to_thread
#     """
#     clean_user_id = user_id.strip()
#     normalized_subjects = [s.strip().lower() for s in subject_list]

#     print(f"--> 🛠️ Processing {len(chunks)} chunks for {clean_user_id}...")
    
#     # CHANGE 2: Call directly (remove 'await asyncio.to_thread')
#     vectors = get_embeddings_batch(chunks)
    
#     if not vectors: return

#     records = []
#     for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
#         if vector is None: continue 

#         vector_id = f"{clean_user_id}_{filename}_{i}"
#         metadata = {
#             "text": chunk,
#             "user_id": clean_user_id,
#             "subject": normalized_subjects,
#             "filename": filename,
#             "type": "personal"
#         }
#         records.append({"id": vector_id, "values": vector, "metadata": metadata})

#     # CHANGE 3: Standard Pinecone Upsert
#     batch_size = 100
#     for i in range(0, len(records), batch_size):
#         batch = records[i:i + batch_size]
#         try:
#             index.upsert(vectors=batch) # Standard blocking call
#             print(f"   -> Batch {i//batch_size + 1} uploaded.")
#         except Exception as e:
#             print(f"   -> ❌ Upload Error: {e}")
# # --- PART C: RETRIEVAL LOGIC (The 2-Layer Logic) ---

# def ask_pinecone(query, user_id_filter=None, subject_filter=None, top_k=5, threshold=0.50):
#     print(f"--- 🔍 Querying Pinecone: User Filter='{user_id_filter}' | Subject='{subject_filter}' ---")
    
#     try:
#         vector = genai.embed_content(
#             model=EMBEDDING_MODEL,
#             content=query,
#             task_type="retrieval_query"
#         )['embedding']
#     except Exception as e:
#         print(f"Embedding Fail: {e}")
#         return []

#     # Build Metadata Filter
#     filter_dict = {}
    
#     # --- CRITICAL FIX FOR STATIC LAYER ---
#     if user_id_filter == "NO_FILTER":
#         # Static Layer: Send EMPTY filter to search untagged 36k records
#         pass 
        
#     else:
#         # Dynamic Layer: Apply strict filters
        
#         # 1. Subject Filter (Only for new data that has tags)
#         if subject_filter:
#             filter_dict['subject'] = {"$in": [subject_filter.strip().lower()]}

#         # 2. User Filter (Only for new data that has tags)
#         if user_id_filter:
#             filter_dict['user_id'] = user_id_filter

#     print(f"--- 🛠️ FINAL FILTER APPLIED: {filter_dict} ---")

#     try:
#         results = index.query(
#             vector=vector,
#             top_k=top_k,
#             include_metadata=True,
#             filter=filter_dict 
#         )
        
#         matches = []
#         for m in results['matches']:
#             # Use the variable 'threshold' instead of hardcoded 0.50
#             if m['score'] > threshold:
#                 matches.append({
#                     'text': m['metadata'].get('text', ''),
#                     'metadata': m['metadata'],
#                     'score': m['score']
#                 })
#         return matches

#     except Exception as e:
#         print(f"❌ Pinecone Query Error: {e}")
#         return []

# import google.generativeai as genai
# from pinecone import Pinecone
# import config
# import asyncio
# from typing import List, Optional

# # Initialize from Config
# genai.configure(api_key=config.GEMINI_API_KEY)

# # Global variables
# pc = None
# index = None

# # Initialize Pinecone
# if config.PINECONE_API_KEY:
#     try:
#         pc = Pinecone(api_key=config.PINECONE_API_KEY)
#         index = pc.Index(config.PINECONE_INDEX_NAME)
#     except Exception as e:
#         print(f"CRITICAL: Pinecone Init Error: {e}")

# async def get_query_embedding_async(text: str) -> Optional[List[float]]:
#     """Generates embedding asynchronously to prevent blocking the server."""
#     try:
#         # Wrap the sync call in to_thread since genai python SDK is primarily sync
#         result = await asyncio.to_thread(
#             genai.embed_content,
#             model="models/text-embedding-004",
#             content=text,
#             task_type="retrieval_query"
#         )
#         return result['embedding']
#     except Exception as e:
#         print(f"Embedding Error: {e}")
#         return None

# def ask_pinecone(query: str, k: int = 5, threshold: float = 0.55, namespace: str = None) -> List[str]:
#     """
#     Returns context strings from Pinecone.
    
#     Args:
#         query: User question
#         k: Number of chunks to retrieve
#         threshold: Minimum similarity score (0.0 to 1.0)
#         namespace: Specific Pinecone namespace to search (optional)
#     """
#     if not index:
#         print("Error: Pinecone index not initialized.")
#         return []
    
#     # If using inside an async route, you would await this
#     # For now, we assume this function is called from a sync context or wrapper
#     vector = get_query_embedding(query) # Use the sync version or await the async one appropriately
    
#     if not vector: 
#         return []

#     try:
#         results = index.query(
#             vector=vector,
#             top_k=k,
#             include_metadata=True,
#             namespace=namespace  # IMPORTANT: Search only specific area if needed
#         )
        
#         context_texts = []
#         for match in results['matches']:
#             # Log scores while debugging to find the sweet spot
#             # print(f"Found match with score: {match['score']}") 
            
#             if match['score'] >= threshold:
#                 meta = match['metadata']
#                 # Robust .get() with default empty string to prevent crashes
#                 source = meta.get('source', 'Unknown Source')
#                 text = meta.get('text', '').strip()
                
#                 if text:
#                     context_texts.append(f"Source: {source} | Content: {text}")
        
#         return context_texts

#     except Exception as e:
#         print(f"Pinecone Search Error: {e}")
#         return []

# # Helper for Sync wrapper if you are not fully async yet
# def get_query_embedding(text):
#     return genai.embed_content(
#         model="models/text-embedding-004",
#         content=text,
#         task_type="retrieval_query"
#     )['embedding']