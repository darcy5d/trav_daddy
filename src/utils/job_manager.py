"""
Background Job Manager for long-running tasks.

Manages async execution of data downloads and model training.
"""

import uuid
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Callable, Optional
from io import StringIO
import sys

logger = logging.getLogger(__name__)

# Global job registry
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


class JobStatus:
    """Job status constants."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class LogCapture:
    """Capture stdout/stderr for a job."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.buffer = StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
    
    def write(self, text):
        """Write to both buffer and original stdout."""
        self.buffer.write(text)
        self.original_stdout.write(text)
        
        # Update job logs
        with _jobs_lock:
            if self.job_id in _jobs:
                _jobs[self.job_id]['logs'].append(text)
    
    def flush(self):
        """Flush output."""
        self.original_stdout.flush()


def start_job(
    name: str,
    func: Callable,
    *args,
    **kwargs
) -> str:
    """
    Start a background job.
    
    Args:
        name: Job name/description
        func: Function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Job ID (UUID)
    """
    job_id = str(uuid.uuid4())
    
    job_info = {
        'id': job_id,
        'name': name,
        'status': JobStatus.PENDING,
        'progress': 0,
        'current_step': None,
        'logs': [],
        'created_at': datetime.now().isoformat(),
        'started_at': None,
        'completed_at': None,
        'error': None,
        'result': None
    }
    
    with _jobs_lock:
        _jobs[job_id] = job_info
    
    # Start thread
    thread = threading.Thread(
        target=_run_job,
        args=(job_id, func, args, kwargs),
        daemon=True
    )
    thread.start()
    
    logger.info(f"Started job {job_id}: {name}")
    return job_id


def _run_job(job_id: str, func: Callable, args: tuple, kwargs: dict):
    """Run a job in a background thread."""
    import threading
    
    try:
        # Store job_id in thread-local storage so functions can access it
        current_thread = threading.current_thread()
        current_thread.job_id = job_id
        
        # Update status to running
        update_job_status(job_id, JobStatus.RUNNING)
        update_job_field(job_id, 'started_at', datetime.now().isoformat())
        
        # Capture logs
        with LogCapture(job_id):
            result = func(*args, **kwargs)
        
        # Update as completed
        update_job_status(job_id, JobStatus.COMPLETED)
        update_job_field(job_id, 'completed_at', datetime.now().isoformat())
        update_job_field(job_id, 'progress', 100)
        update_job_field(job_id, 'result', result)
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()
        update_job_status(job_id, JobStatus.FAILED)
        update_job_field(job_id, 'error', str(e))
        update_job_field(job_id, 'completed_at', datetime.now().isoformat())


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get status of a job.
    
    Args:
        job_id: Job ID
        
    Returns:
        Job info dictionary or None if not found
    """
    with _jobs_lock:
        return _jobs.get(job_id)


def update_job_status(job_id: str, status: str):
    """Update job status."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]['status'] = status


def update_job_progress(job_id: str, progress: int, current_step: Optional[str] = None):
    """
    Update job progress.
    
    Args:
        job_id: Job ID
        progress: Progress percentage (0-100)
        current_step: Optional description of current step
    """
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]['progress'] = progress
            if current_step:
                _jobs[job_id]['current_step'] = current_step


def update_job_field(job_id: str, field: str, value: Any):
    """Update a specific field in the job info."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id][field] = value


def get_all_jobs() -> Dict[str, Dict[str, Any]]:
    """Get all jobs."""
    with _jobs_lock:
        return dict(_jobs)


def clear_completed_jobs(max_age_seconds: int = 3600):
    """
    Clear completed/failed jobs older than max_age_seconds.
    
    Args:
        max_age_seconds: Maximum age in seconds (default 1 hour)
    """
    current_time = time.time()
    
    with _jobs_lock:
        to_remove = []
        for job_id, job_info in _jobs.items():
            if job_info['status'] in [JobStatus.COMPLETED, JobStatus.FAILED]:
                if job_info['completed_at']:
                    completed_time = datetime.fromisoformat(job_info['completed_at']).timestamp()
                    if current_time - completed_time > max_age_seconds:
                        to_remove.append(job_id)
        
        for job_id in to_remove:
            del _jobs[job_id]
            logger.debug(f"Removed old job {job_id}")

