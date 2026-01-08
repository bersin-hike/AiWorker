##########WORKER.PY##############
"""
# Background Worker Process for PDF Processing
# Runs independently from the web server to avoid GIL blocking.
# """
import logging
import sys
import os
import json
import time
import redis
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
# Custom imports
from processor import process_file, chunk_text
from query_logic import add_to_vector_db
# Load environment
load_dotenv()
# Setup Logging
logger = logging.getLogger("PDFWorker")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler(sys.stdout)
c_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
f_handler = RotatingFileHandler('worker.log', maxBytes=10_000_000, backupCount=5, encoding='utf-8')
f_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s'))
logger.addHandler(c_handler)
logger.addHandler(f_handler)
logger.info("🚀 PDF Worker Process Starting...")
# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    logger.error("❌ REDIS_URL not found in environment. Worker cannot start.")
    sys.exit(1)
def get_redis_connection():
    """Create a Redis connection with retry logic."""
    try:
        r_client = redis.from_url(REDIS_URL, decode_responses=True)
        r_client.ping()
        logger.info("✅ Redis connection established.")
        return r_client
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return None
def worker_loop():
    """
    Main worker loop: BLPOP from queue, process PDF, update status.
    """
    r_client = get_redis_connection()
    
    if not r_client:
        logger.error("❌ Cannot start worker without Redis. Exiting.")
        sys.exit(1)
    
    logger.info("🔄 Worker is ready. Listening for jobs...")
    
    while True:
        try:
            # 1. Blocking Pop with 5-second timeout
            # Format: BLPOP returns (queue_name, item) or None
            result = r_client.blpop("pdf_queue", timeout=5)
            
            if not result:
                # Timeout - no job available. Loop again.
                continue
            
            # 2. Parse Job
            _, packed_item = result
            task = json.loads(packed_item)
            
            filename = task.get('filename')
            file_path = task.get('path')
            user_id = task.get('user_id')
            subject_list = task.get('subjects', [])
            
            logger.info(f"📦 Job Received: {filename}")
            
            # 3. Update Status to Processing
            r_client.set(f"status:{filename}", "processing", ex=86400)
            
            # 4. Process the PDF
            try:
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        raw_text = process_file(f, filename)
                        
                        if raw_text:
                            chunks = chunk_text(raw_text)
                            add_to_vector_db(chunks, user_id, subject_list, filename)
                            logger.info(f"✅ Successfully processed: {filename}")
                            r_client.set(f"status:{filename}", "completed", ex=86400)
                        else:
                            logger.warning(f"⚠️ Empty content: {filename}")
                            r_client.set(f"status:{filename}", "completed", ex=86400)
                    
                    # 5. Cleanup: Delete the file from disk
                    try:
                        os.remove(file_path)
                        logger.info(f"🗑️ Deleted: {filename}")
                    except Exception as cleanup_err:
                        logger.error(f"❌ Failed to delete {filename}: {cleanup_err}")
                else:
                    logger.error(f"❌ File not found: {filename}")
                    r_client.set(f"status:{filename}", "failed", ex=86400)
                    
            except Exception as job_error:
                logger.error(f"❌ Processing error for {filename}: {job_error}", exc_info=True)
                r_client.set(f"status:{filename}", "failed", ex=86400)
        
        except redis.ConnectionError as conn_err:
            logger.error(f"❌ Redis connection lost: {conn_err}")
            time.sleep(5)
            r_client = get_redis_connection()
            if not r_client:
                logger.error("❌ Could not reconnect to Redis. Exiting.")
                sys.exit(1)
        
        except Exception as e:
            logger.error(f"❌ Unexpected error in worker loop: {e}", exc_info=True)
            time.sleep(2)
if __name__ == "__main__":
    logger.info("🎯 Starting worker loop...")
    worker_loop()
