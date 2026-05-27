import logging
from redis import Redis

logger = logging.getLogger(__name__)

def delete_old_threads(redis_client: Redis, max_age_seconds: int):
    """
    Deletes Redis keys associated with LangGraph threads that have been idle
    for longer than max_age_seconds.
    """
    keys_to_delete = set()
    
    # LangGraph RedisSaver uses patterns like 'checkpoint:<thread_id>:...'
    # and 'writes:<thread_id>:...'
    for key in redis_client.scan_iter("checkpoint:*"):
        idle_time = redis_client.object("idletime", key)
        
        if idle_time is not None and idle_time > max_age_seconds:
            # Extract thread_id: 'checkpoint:thread_id:ns' -> thread_id is index 1
            parts = key.split(":")
            if len(parts) >= 2:
                thread_id = parts[1]
                # Find and queue all keys associated with this specific thread_id
                for related_key in redis_client.scan_iter(f"*:{thread_id}:*"):
                    keys_to_delete.add(related_key)
                keys_to_delete.add(key)

    if keys_to_delete:
        redis_client.delete(*keys_to_delete)
        logger.info(f"Cleaned up {len(keys_to_delete)} keys associated with stale threads (>{max_age_seconds}s idle).")
    else:
        logger.info("No stale threads found. Redis is clean.")