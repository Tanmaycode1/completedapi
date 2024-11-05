from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import logging
import os
from datetime import datetime, timedelta
import time
import json
import tempfile
import uuid
import shutil
import psutil
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pypdf
from pdf2image import convert_from_path
import io
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Constants for PDF Processing
TEMP_DIR = "/tmp/pdf_processing"
DPI = 300  # Optimized DPI setting
RESIZE_FACTOR = 1.5  # Optimized resize factor
CHUNK_SIZE = 3  # Number of pages to process in parallel
COMPRESSION_QUALITY = 85  # Image compression quality
MAX_THREADS = min(4, os.cpu_count() or 2)
SESSION_TIMEOUT = 3600
MAX_MEMORY_PERCENT = 80
CLEANUP_INTERVAL = 300

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
pdf_tasks: Dict[str, Dict] = {}
active_sessions: Dict[str, datetime] = {}
session_locks: Dict[str, threading.Lock] = {}

class PDFProcessingRequest(BaseModel):
    prompt_type: str = Field(
        default="policy_json_conversion",
        description="Type of processing to apply to the PDF"
    )
    priority: Optional[int] = Field(
        default=3,
        ge=1,
        le=5,
        description="Processing priority (1-5)"
    )
    max_tokens: Optional[int] = Field(
        default=4000,
        ge=1,
        le=8000,
        description="Maximum tokens for processing"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for text generation"
    )
    ocr_language: str = Field(
        default="eng",
        description="Language for OCR processing"
    )
    enhance_scan: bool = Field(
        default=True,
        description="Apply image enhancement for better OCR results"
    )

async def optimize_image_for_ocr(image: Image.Image) -> np.ndarray:
    """Optimized image preprocessing with better performance"""
    # Convert to numpy array once
    img_np = np.array(image)
    
    # Convert to grayscale if needed (faster check)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    
    # Resize more efficiently
    height, width = gray.shape
    new_height = int(height * RESIZE_FACTOR)
    new_width = int(width * RESIZE_FACTOR)
    scaled = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Apply optimized binary threshold
    binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Quick denoise (faster than Non-local Means)
    denoised = cv2.medianBlur(binary, 3)
    
    return denoised

def process_page_chunk(chunk: List[Tuple[int, Image.Image]], lang: str) -> List[Tuple[int, str]]:
    """Process a chunk of pages in parallel"""
    results = []
    
    for page_num, image in chunk:
        try:
            # Convert PIL Image to bytes in memory
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG', optimize=True, quality=COMPRESSION_QUALITY)
            img_byte_arr = img_byte_arr.getvalue()
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(img_byte_arr, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Quick image optimization
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Use simpler tesseract configuration for speed
            text = pytesseract.image_to_string(
                img,
                lang=lang,
                config='--oem 3 --psm 6'
            )
            
            results.append((page_num, text))
            
        except Exception as e:
            results.append((page_num, f"Error processing page {page_num}: {str(e)}"))
    
    return results

async def extract_text_from_pages(images: List[Image.Image], lang: str = 'eng') -> List[Tuple[int, str]]:
    """Extract text from multiple pages using parallel processing"""
    # Create page chunks for batch processing
    page_chunks = []
    current_chunk = []
    
    for i, image in enumerate(images, 1):
        current_chunk.append((i, image))
        if len(current_chunk) == CHUNK_SIZE:
            page_chunks.append(current_chunk)
            current_chunk = []
    
    if current_chunk:
        page_chunks.append(current_chunk)
    
    # Process chunks in parallel
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
        futures = []
        for chunk in page_chunks:
            future = loop.run_in_executor(
                executor,
                partial(process_page_chunk, chunk, lang)
            )
            futures.append(future)
        
        # Gather results
        results = []
        for future in await asyncio.gather(*futures):
            results.extend(future)
    
    return sorted(results, key=lambda x: x[0])

async def process_pdf_content(
    content: bytes,
    task_id: str,
    session_id: str,
    prompt_type: str,
    ocr_language: str = 'eng',
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
):
    """Optimized PDF processing function"""
    session_dir = os.path.join(TEMP_DIR, session_id)
    temp_path = os.path.join(session_dir, task_id)
    os.makedirs(temp_path, exist_ok=True)
    pdf_path = os.path.join(temp_path, "input.pdf")
    
    try:
        # Save PDF
        with open(pdf_path, 'wb') as f:
            f.write(content)
        
        # Try direct text extraction first
        text_content = []
        needs_ocr = True
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text and len(text.split()) > 5:
                        text_content.append(f"\nPage {page_num}\n{'='*50}\n\n{text}")
                
                if len(text_content) >= (len(pdf_reader.pages) * 0.5):
                    needs_ocr = False
                    final_text = '\n'.join(text_content)
                    pdf_tasks[task_id].update({
                        'status': 'completed',
                        'result': final_text,
                        'completion_time': datetime.utcnow().isoformat(),
                        'ocr_used': False
                    })
                    return
                    
        except Exception as e:
            logger.error(f"Direct text extraction failed: {str(e)}")
        
        if needs_ocr:
            # Convert PDF to images with optimized settings
            images = convert_from_path(
                pdf_path,
                dpi=DPI,
                output_folder=temp_path,
                fmt='png',
                grayscale=True,
                thread_count=min(os.cpu_count(), 4),
                use_pdftocairo=True
            )
            
            total_pages = len(images)
            logger.info(f"Processing {total_pages} pages for task {task_id}")
            
            # Update task with total pages
            pdf_tasks[task_id].update({
                'total_pages': total_pages,
                'pages_processed': 0
            })
            
            # Process pages in parallel
            results = await extract_text_from_pages(images, ocr_language)
            
            # Combine results
            text_parts = [f"Document Information:\nTotal Pages: {total_pages}\n"]
            for page_num, text in results:
                if text.strip():
                    text_parts.append(f"\nPage {page_num}\n{'='*50}\n")
                    text_parts.append(text)
                
                # Update progress
                pdf_tasks[task_id].update({
                    'progress': (page_num / total_pages) * 100,
                    'pages_processed': page_num
                })
            
            final_text = '\n'.join(text_parts)
            
            pdf_tasks[task_id].update({
                'status': 'completed',
                'result': final_text,
                'completion_time': datetime.utcnow().isoformat(),
                'ocr_used': True,
                'word_count': len(final_text.split())
            })
            
    except Exception as e:
        logger.error(f"Error processing PDF {task_id}: {str(e)}")
        pdf_tasks[task_id].update({
            'status': 'failed',
            'error': str(e),
            'error_time': datetime.utcnow().isoformat()
        })
    finally:
        try:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")
        gc.collect()

@app.post("/api/process-pdf")
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: PDFProcessingRequest = Depends()
):
    """Enhanced PDF processing endpoint with better error handling and progress tracking"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Only PDF files are supported."
        )

    try:
        # Create new session and task IDs
        session_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        
        # Initialize session
        active_sessions[session_id] = datetime.utcnow()
        session_locks[session_id] = threading.Lock()
        
        # Read file content
        content = await file.read()
        
        # Validate PDF content
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty PDF file provided"
            )
            
        # Initialize task with detailed metadata
        pdf_tasks[task_id] = {
            'status': 'processing',
            'filename': file.filename,
            'session_id': session_id,
            'prompt_type': request.prompt_type,
            'submission_time': datetime.utcnow().isoformat(),
            'priority': request.priority,
            'ocr_language': request.ocr_language,
            'enhance_scan': request.enhance_scan,
            'file_size': len(content),
            'progress': 0,
            'pages_processed': 0,
            'total_pages': None
        }

        # Start background processing
        background_tasks.add_task(
            process_pdf_content,
            content,
            task_id,
            session_id,
            request.prompt_type,
            request.ocr_language,
            request.max_tokens,
            request.temperature
        )

        return {
            'task_id': task_id,
            'session_id': session_id,
            'status': 'processing',
            'message': 'PDF processing started',
            'estimated_time': 'Calculating...',
            'filename': file.filename
        }

    except Exception as e:
        logger.error(f"Error initiating PDF processing: {str(e)}")
        if 'session_id' in locals():
            await session_manager.cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pdf-status/{task_id}")
async def get_pdf_status(task_id: str):
    """Enhanced status endpoint with detailed progress information"""
    if task_id not in pdf_tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found. The task may have expired or been cleaned up."
        )
    
    task = pdf_tasks[task_id]
    
    # Update session access time
    session_id = task.get('session_id')
    if session_id and session_id in active_sessions:
        active_sessions[session_id] = datetime.utcnow()
    
    # Calculate processing time
    start_time = datetime.fromisoformat(task['submission_time'])
    processing_time = (datetime.utcnow() - start_time).total_seconds()
    
    response = {
        'status': task['status'],
        'filename': task.get('filename'),
        'processing_time': round(processing_time, 2),
        'progress': task.get('progress', 0),
        'pages_processed': task.get('pages_processed', 0),
        'total_pages': task.get('total_pages'),
        'ocr_used': task.get('ocr_used', False),
    }
    
    if task['status'] == 'completed':
        response.update({
            'text_content': task['result'],
            'completion_time': task.get('completion_time'),
            'text_length': len(task['result']),
            'word_count': len(task['result'].split()),
        })
    elif task['status'] == 'failed':
        response.update({
            'error': task.get('error'),
            'error_time': task.get('error_time'),
            'last_successful_page': task.get('last_successful_page', 0)
        })
    
    return response


class SessionManager:
    def __init__(self):
        self.session_cleanup_task = None
        self.memory_monitor_task = None
        
    async def start(self):
        """Start background tasks for session and memory management"""
        self.session_cleanup_task = asyncio.create_task(self.cleanup_sessions())
        self.memory_monitor_task = asyncio.create_task(self.monitor_memory())
        
    async def cleanup_sessions(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_sessions = [
                    session_id for session_id, last_access 
                    in active_sessions.items()
                    if (current_time - last_access).total_seconds() > SESSION_TIMEOUT
                ]
                
                for session_id in expired_sessions:
                    await self.cleanup_session(session_id)
                    
                # Force garbage collection after cleanup
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {str(e)}")
                
            await asyncio.sleep(CLEANUP_INTERVAL)
    
    async def monitor_memory(self):
        """Monitor system memory usage and trigger cleanup if needed"""
        while True:
            try:
                memory = psutil.virtual_memory()
                if memory.percent > MAX_MEMORY_PERCENT:
                    logger.warning(f"High memory usage detected: {memory.percent}%")
                    # Force cleanup of oldest sessions until memory usage is acceptable
                    while memory.percent > MAX_MEMORY_PERCENT and active_sessions:
                        oldest_session = min(active_sessions.items(), key=lambda x: x[1])[0]
                        await self.cleanup_session(oldest_session)
                        memory = psutil.virtual_memory()
                    
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error in memory monitoring: {str(e)}")
                
            await asyncio.sleep(60)  # Check every minute
            
    async def cleanup_session(self, session_id: str):
        """Clean up a specific session and its resources"""
        try:
            # Acquire session lock
            lock = session_locks.get(session_id)
            if lock:
                lock.acquire()
                
            # Clean up session files
            session_dir = os.path.join(TEMP_DIR, session_id)
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
                
            # Remove session data
            if session_id in pdf_tasks:
                del pdf_tasks[session_id]
            if session_id in active_sessions:
                del active_sessions[session_id]
            if session_id in session_locks:
                del session_locks[session_id]
                
            logger.info(f"Cleaned up session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {str(e)}")
            
        finally:
            if lock:
                lock.release()
                
    async def get_active_sessions(self):
        """Get count and details of active sessions"""
        try:
            current_time = datetime.utcnow()
            active_count = len(active_sessions)
            session_details = [
                {
                    'session_id': session_id,
                    'age': (current_time - last_access).total_seconds(),
                    'task_count': len([t for t in pdf_tasks.values() if t.get('session_id') == session_id])
                }
                for session_id, last_access in active_sessions.items()
            ]
            
            return {
                'active_count': active_count,
                'sessions': session_details
            }
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {str(e)}")
            return {'active_count': 0, 'sessions': []}

# Initialize session manager
session_manager = SessionManager()

# Add startup event to initialize session manager
@app.on_event("startup")
async def startup_event():
    """Initialize application resources on startup"""
    try:
        # Create temporary directory
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Start session manager
        await session_manager.start()
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

# Add shutdown event for cleanup
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    try:
        # Clean up all sessions
        for session_id in list(active_sessions.keys()):
            await session_manager.cleanup_session(session_id)
        
        # Clean up temporary directory
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            
        logger.info("Application shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


@app.post("/api/generate-encoding")
async def generate_encoding(request: Dict[str, Any]):
    try:
        req = RequestModel(**request)
        config = get_prompt_config("encoding")
        messages = [
            {"role": "system", "content": config.system_content},
            {"role": "user", "content": req.body}
        ]

        return await openai_client.get_complete_response(
            messages,
            config,
            req.max_tokens,
            req.temperature
        )
    except Exception as e:
        logger.error(f"Error in generate_encoding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))






# Session handling endpoints
sessions = {}


@app.post("/api/start-session")
async def start_session(request: Dict[str, Any]):
    try:
        session_id = str(os.urandom(16).hex())
        sessions[session_id] = {
            "context": [
                {
                    "role": "system",
                    "content": "You are an assistant specializing in prior authorization for basys.ai. "
                               "Your responses should be based exclusively on the provided policy encoding and patient data."
                }
            ],
            "last_access": time.time()
        }

        if "body" in request:
            sessions[session_id]["context"].append({
                "role": "user",
                "content": request["body"]
            })

        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Error in start_session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/send-message")
async def send_message(request: Dict[str, Any]):
    try:
        if "session_id" not in request or "message" not in request:
            raise HTTPException(status_code=400, detail="Missing session_id or message")

        session_id = request["session_id"]
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        # Update session context with new message
        sessions[session_id]["context"].append({
            "role": "user",
            "content": request["message"]
        })

        # Generate response
        messages = sessions[session_id]["context"]
        config = get_prompt_config("encoding")  # Using default config for sessions

        response = await openai_client.get_complete_response(
            messages,
            config,
            4000,  # Default max tokens
            0.7  # Default temperature
        )

        response_content = response if isinstance(response, str) else json.dumps(response)

        # Update session context with assistant's response
        sessions[session_id]["context"].append({
            "role": "assistant",
            "content": response_content
        })

        sessions[session_id]["last_access"] = time.time()

        return {"response": response_content}

    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "openai_client": "initialized",
        "sessions_active": len(sessions),
        "pdf_tasks": {
            "active": len(pdf_tasks),
            "completed": len([t for t in pdf_tasks.values() if t['status'] == 'completed']),
            "failed": len([t for t in pdf_tasks.values() if t['status'] == 'failed'])
        }
    }

