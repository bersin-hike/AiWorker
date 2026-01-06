###########implemnting  Ai-intro feature##############
"""
Background Worker Process for PDF Processing
"""
import logging
import sys
import os
import json
import time
import redis
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from processor import process_file, chunk_text
from query_logic import add_to_vector_db

load_dotenv()
logger = logging.getLogger("PDFWorker")
logger.setLevel(logging.INFO)

c_handler = logging.StreamHandler(sys.stdout)
c_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(c_handler)

logger.info("🚀 PDF Worker Process Starting...")
REDIS_URL = os.getenv("RENDER_REDIS_URL") 

if not REDIS_URL:
    logger.error("❌ RENDER_REDIS_URL not found. Worker cannot start.")
    sys.exit(1)

def get_redis_connection():
    """Create a Redis connection with retry logic."""
    try:
        r_client = redis.from_url(REDIS_URL, decode_responses=True)
        r_client.ping()
        logger.info("✅ Connected to Render Queue Redis.")
        return r_client
    except Exception as e:
        logger.error(f"❌ Redis Init Failed: {e}")
        return None

def worker_loop():
    r_client = get_redis_connection()
    
    if not r_client:
        sys.exit(1)
    
    logger.info("🔄 Worker is ready. Listening for jobs on 'pdf_queue'...")
    
    while True:
        try:
            result = r_client.blpop("pdf_queue", timeout=5)
            
            if not result:
                continue 
            
            _, packed_item = result
            task = json.loads(packed_item)
            
            filename = task.get('filename')
            file_path = task.get('path')
            user_id = task.get('user_id')
            subjects = task.get('subjects', [])
            
            logger.info(f"📦 Processing: {filename}")
            r_client.set(f"status:{filename}", "processing", ex=86400)
            
            try:
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        raw_text = process_file(f, filename)
                        
                        if raw_text:
                            chunks = chunk_text(raw_text)
                            add_to_vector_db(chunks, user_id, subjects, filename)
                            logger.info(f"✅ Finished: {filename}")
                            r_client.set(f"status:{filename}", "completed", ex=86400)
                        else:
                            logger.warning(f"⚠️ File was empty: {filename}")
                            r_client.set(f"status:{filename}", "completed", ex=86400)
                    
                    os.remove(file_path)
                else:
                    logger.error(f"❌ File missing from disk: {filename}")
                    r_client.set(f"status:{filename}", "failed", ex=86400)
                    
            except Exception as job_error:
                logger.error(f"❌ Logic Error: {job_error}", exc_info=True)
                r_client.set(f"status:{filename}", "failed", ex=86400)
        
        except redis.ConnectionError:
            logger.warning("⚠️ Redis connection lost. Reconnecting in 5s...")
            time.sleep(5)
            r_client = get_redis_connection()
            
        except Exception as e:
            logger.error(f"❌ Critical Worker Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    worker_loop()