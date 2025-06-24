#!/usr/bin/env python3
#bridge.py
"""
Bridge Server for Dosu Intelligence UI

This bridge connects the HTML frontend with the retriever backend:
- Serves the static HTML UI
- Manages chat sessions with JSON persistence 
- Handles real-time streaming of retriever events
- Proxies API calls to the retriever FastAPI service
- Maintains chat history and session state

Author: Praveen
License: MIT
"""

import json
import os
import asyncio
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

# FastAPI and web imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# HTTP client for retriever API calls
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ChatMessage(BaseModel):
    id: str
    type: str  # "user" or "assistant"
    content: str
    timestamp: str
    session_id: str

class ChatSession(BaseModel):
    id: str
    title: str
    created_at: str
    last_message_at: str
    messages: List[ChatMessage]
    attached_sources: List[str] = []

class QueryRequest(BaseModel):
    query: str
    session_id: str

class NewSessionRequest(BaseModel):
    title: Optional[str] = None

class AttachSourceRequest(BaseModel):
    session_id: str
    source_name: str

class BridgeServer:
    """Main bridge server class"""
    
    def __init__(self, retriever_url: str = "http://localhost:8001", data_dir: str = "./data"):
        self.retriever_url = retriever_url
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Session storage
        self.sessions_file = self.data_dir / "sessions.json"
        self.sessions: Dict[str, ChatSession] = {}
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Load existing sessions
        self._load_sessions()
        
        # HTTP client for retriever API
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        logger.info(f"Bridge server initialized - Retriever: {retriever_url}, Data: {data_dir}")
    
    def _load_sessions(self):
        """Load sessions from JSON file"""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    sessions_data = json.load(f)
                    for session_data in sessions_data.get('sessions', []):
                        session = ChatSession(**session_data)
                        self.sessions[session.id] = session
                logger.info(f"Loaded {len(self.sessions)} existing sessions")
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
            self.sessions = {}
    
    def _save_sessions(self):
        """Save sessions to JSON file"""
        try:
            sessions_data = {
                "sessions": [session.model_dump() for session in self.sessions.values()],
                "last_updated": datetime.now().isoformat()
            }
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
    
    def create_session(self, title: str = None) -> ChatSession:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        if not title:
            title = f"New Chat"
        
        session = ChatSession(
            id=session_id,
            title=title,
            created_at=timestamp,
            last_message_at=timestamp,
            messages=[],
            attached_sources=[]
        )
        
        self.sessions[session_id] = session
        self._save_sessions()
        
        logger.info(f"Created new session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def add_message(self, session_id: str, message_type: str, content: str) -> ChatMessage:
        """Add message to session"""
        session = self.sessions.get(session_id)
        if not session:
            # Auto-create session if it doesn't exist
            logger.info(f"Auto-creating session for message: {session_id}")
            session = ChatSession(
                id=session_id,
                title="New Chat",
                created_at=datetime.now().isoformat(),
                last_message_at=datetime.now().isoformat(),
                messages=[],
                attached_sources=[]
            )
            self.sessions[session_id] = session
        
        message = ChatMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            content=content,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
        session.messages.append(message)
        session.last_message_at = message.timestamp
        
        # Update session title if this is the first user message
        if message_type == "user" and len([m for m in session.messages if m.type == "user"]) == 1:
            session.title = content[:50] + "..." if len(content) > 50 else content
        
        self._save_sessions()
        return message
    
    def attach_source(self, session_id: str, source_name: str):
        """Attach a source to session"""
        session = self.sessions.get(session_id)
        if not session:
            # Auto-create session if it doesn't exist
            logger.info(f"Auto-creating session for source attachment: {session_id}")
            session = ChatSession(
                id=session_id,
                title="New Chat",
                created_at=datetime.now().isoformat(),
                last_message_at=datetime.now().isoformat(),
                messages=[],
                attached_sources=[]
            )
            self.sessions[session_id] = session
        
        if source_name not in session.attached_sources:
            session.attached_sources.append(source_name)
            self._save_sessions()
        
        logger.info(f"Attached source {source_name} to session {session_id}")
    
    async def process_query(self, session_id: str, query: str) -> Dict:
        """Process query through retriever API"""
        try:
            # Auto-create session if it doesn't exist
            if session_id not in self.sessions:
                logger.info(f"Auto-creating session: {session_id}")
                session = ChatSession(
                    id=session_id,
                    title="New Chat",
                    created_at=datetime.now().isoformat(),
                    last_message_at=datetime.now().isoformat(),
                    messages=[],
                    attached_sources=[]
                )
                self.sessions[session_id] = session
                self._save_sessions()
            
            # Add user message
            user_message = self.add_message(session_id, "user", query)
            
            # Send to retriever API - Use client directly, not with async with
            response = await self.http_client.post(
                f"{self.retriever_url}/query",
                json={"query": query, "session_id": session_id},
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()
            
            # Add assistant response
            assistant_message = self.add_message(session_id, "assistant", result["response"])
            
            return {
                "user_message": user_message.model_dump(),
                "assistant_message": assistant_message.model_dump(),
                "debug_info": result.get("debug_info", {}),
                "session": self.sessions[session_id].model_dump()
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Ensure session exists for error message
            if session_id not in self.sessions:
                session = ChatSession(
                    id=session_id,
                    title="New Chat",
                    created_at=datetime.now().isoformat(),
                    last_message_at=datetime.now().isoformat(),
                    messages=[],
                    attached_sources=[]
                )
                self.sessions[session_id] = session
                self._save_sessions()
            
            error_message = self.add_message(session_id, "assistant", f"Sorry, I encountered an error: {str(e)}")
            return {
                "user_message": user_message.model_dump() if 'user_message' in locals() else None,
                "assistant_message": error_message.model_dump(),
                "error": str(e),
                "session": self.sessions[session_id].model_dump()
            }

# Initialize FastAPI app
app = FastAPI(title="Dosu Intelligence Bridge", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize bridge server
bridge = BridgeServer()

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Dosu Intelligence Bridge Server starting up...")
    
    # Test connection to retriever
    try:
        response = await bridge.http_client.get(f"{bridge.retriever_url}/health", timeout=5.0)
        if response.status_code == 200:
            logger.info("✅ Connected to retriever API")
        else:
            logger.warning(f"⚠️ Retriever API responded with status {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to retriever API: {e}")

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    await bridge.http_client.aclose()
    logger.info("🛑 Bridge server shutdown complete")

# Serve static files (HTML UI) - EXACT ORIGINAL CODE
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main UI"""
    # Return the EXACT ORIGINAL HTML content
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dosu Intelligence - Code Intelligence Reimagined</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #000000 0%, #0f0f23 50%, #1a1a2e 100%);
            color: white;
            overflow: hidden;
            height: 100vh;
            position: relative;
        }

        /* Animated background particles */
        .bg-particles {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            overflow: hidden;
        }

        .particle {
            position: absolute;
            width: 2px;
            height: 2px;
            background: rgba(59, 130, 246, 0.2);
            border-radius: 50%;
            animation: float 8s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.2; }
            50% { transform: translateY(-25px) rotate(180deg); opacity: 0.6; }
        }

        /* Main container */
        .app-container {
            display: flex;
            height: 100vh;
            position: relative;
            z-index: 1;
        }

        /* Left Sidebar */
        .sidebar {
            width: 280px;
            display: flex;
            flex-direction: column;
            background: rgba(8, 8, 8, 0.8);
            backdrop-filter: blur(20px) saturate(180%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            position: relative;
            overflow: hidden;
        }

        .sidebar::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.4), transparent);
        }

        /* Top Header with Org Selection */
        .sidebar-header {
            padding: 20px 24px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            background: rgba(0, 0, 0, 0.3);
        }

        .org-selector {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 12px 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            margin-bottom: 16px;
        }

        .org-selector:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(59, 130, 246, 0.3);
            transform: translateY(-1px);
        }

        .org-selector-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .org-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .org-avatar {
            width: 28px;
            height: 28px;
            border-radius: 6px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 600;
            color: white;
        }

        .org-details h3 {
            font-size: 14px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 2px;
        }

        .org-details p {
            font-size: 12px;
            color: rgba(255, 255, 255, 0.5);
        }

        .dropdown-arrow {
            color: rgba(255, 255, 255, 0.4);
            transition: transform 0.3s ease;
        }

        .personal-selector {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 12px 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }

        .personal-selector:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(59, 130, 246, 0.3);
            transform: translateY(-1px);
        }

        .personal-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .personal-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .personal-icon {
            width: 28px;
            height: 28px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            color: rgba(255, 255, 255, 0.7);
        }

        /* Navigation Tabs */
        .nav-tabs {
            padding: 20px 24px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            flex: 1;
        }

        .nav-tab {
            padding: 16px 20px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: rgba(255, 255, 255, 0.7);
            font-size: 15px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .nav-tab::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(147, 51, 234, 0.1));
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .nav-tab:hover::before {
            opacity: 1;
        }

        .nav-tab:hover {
            background: rgba(255, 255, 255, 0.08);
            color: rgba(255, 255, 255, 0.9);
            border-color: rgba(59, 130, 246, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
        }

        .nav-tab.active {
            background: rgba(59, 130, 246, 0.12);
            border-color: rgba(59, 130, 246, 0.4);
            color: #60a5fa;
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
        }

        .nav-tab.active::before {
            opacity: 1;
        }

        .nav-icon {
            font-size: 18px;
            width: 20px;
            text-align: center;
        }

        .nav-label {
            flex: 1;
        }

        .nav-badge {
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            padding: 2px 8px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 600;
        }

        /* Sources Section at Bottom */
        .sources-section {
            padding: 24px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            background: rgba(0, 0, 0, 0.4);
        }

        .sources-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }

        .github-logo {
            width: 20px;
            height: 20px;
            fill: rgba(255, 255, 255, 0.6);
        }

        .sources-title {
            font-size: 15px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.9);
            letter-spacing: -0.02em;
        }

        /* Connected Sources Container */
        .sources-container {
            position: relative;
            transition: all 0.3s ease;
        }

        .sources-preview {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(15px);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            overflow: hidden;
        }

        .sources-preview::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(147, 51, 234, 0.08));
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .sources-preview:hover::before {
            opacity: 1;
        }

        .sources-preview:hover {
            transform: translateY(-2px);
            border-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 15px 30px rgba(59, 130, 246, 0.1);
        }

        .preview-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }

        .preview-title {
            font-size: 14px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.9);
        }

        .repo-count {
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            padding: 4px 10px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
        }

        .preview-repos {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }

        .repo-avatar {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 600;
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.1);
        }

        .expand-hint {
            color: rgba(255, 255, 255, 0.4);
            font-size: 12px;
            text-align: center;
        }

        /* Expanded Sources */
        .sources-expanded {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(5, 5, 5, 0.98);
            backdrop-filter: blur(30px);
            border-radius: 20px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            opacity: 0;
            transform: translateY(20px);
            pointer-events: none;
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            z-index: 10;
            max-height: 300px;
        }

        .sources-container:hover .sources-expanded {
            opacity: 1;
            transform: translateY(0);
            pointer-events: all;
        }

        .sources-container:hover .sources-preview {
            opacity: 0;
            transform: scale(0.95);
        }

        .scroll-indicators {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-bottom: 16px;
            opacity: 0.5;
        }

        .scroll-arrow {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            color: rgba(255, 255, 255, 0.6);
        }

        .repos-list {
            max-height: 200px;
            overflow-y: auto;
            padding-right: 8px;
        }

        .repo-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            cursor: grab;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            overflow: hidden;
        }

        .repo-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .repo-card:hover {
            transform: translateY(-4px);
            border-color: rgba(59, 130, 246, 0.4);
            box-shadow: 0 15px 30px rgba(59, 130, 246, 0.15);
        }

        .repo-card:hover::before {
            opacity: 1;
        }

        .repo-card:active {
            cursor: grabbing;
            transform: scale(1.05) rotate(2deg);
        }

        .repo-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .repo-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 600;
            color: white;
        }

        .repo-details h4 {
            font-size: 14px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 4px;
        }

        .repo-details p {
            font-size: 12px;
            color: rgba(255, 255, 255, 0.5);
        }

        /* Middle Panel - Chat History */
        .chat-history-panel {
            width: 320px;
            background: rgba(8, 8, 8, 0.9);
            backdrop-filter: blur(25px) saturate(180%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            flex-direction: column;
            position: relative;
            transition: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        }

        .chat-history-panel.collapsed {
            width: 0;
            opacity: 0;
            pointer-events: none;
        }

        .history-panel-header {
            padding: 24px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            background: rgba(0, 0, 0, 0.4);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .history-panel-title {
            font-size: 18px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.95);
            letter-spacing: -0.02em;
        }

        .panel-controls {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .new-chat-btn {
            background: rgba(59, 130, 246, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 10px;
            padding: 8px 12px;
            color: #60a5fa;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .new-chat-btn:hover {
            background: rgba(59, 130, 246, 0.25);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        }

        .collapse-btn {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 8px;
            color: rgba(255, 255, 255, 0.6);
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
        }

        .collapse-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            color: rgba(255, 255, 255, 0.9);
        }

        .search-box {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 12px 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin: 16px 24px;
            display: flex;
            align-items: center;
            gap: 12px;
            transition: all 0.3s ease;
        }

        .search-box:focus-within {
            border-color: rgba(59, 130, 246, 0.4);
            background: rgba(255, 255, 255, 0.06);
        }

        .search-icon {
            color: rgba(255, 255, 255, 0.4);
            font-size: 14px;
        }

        .search-input {
            flex: 1;
            background: transparent;
            border: none;
            color: rgba(255, 255, 255, 0.9);
            font-size: 14px;
            outline: none;
            font-family: inherit;
        }

        .search-input::placeholder {
            color: rgba(255, 255, 255, 0.4);
        }

        .history-list {
            flex: 1;
            overflow-y: auto;
            padding: 0 24px 24px;
        }

        .history-item {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            overflow: hidden;
        }

        .history-item::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(180deg, #3b82f6, #8b5cf6);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .history-item:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(59, 130, 246, 0.2);
            transform: translateX(4px);
        }

        .history-item:hover::before {
            opacity: 1;
        }

        .history-item.active {
            background: rgba(59, 130, 246, 0.08);
            border-color: rgba(59, 130, 246, 0.3);
        }

        .history-item.active::before {
            opacity: 1;
        }

        .history-title-text {
            font-size: 14px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 6px;
            line-height: 1.4;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .history-preview {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.5);
            line-height: 1.3;
            margin-bottom: 12px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .history-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .history-time {
            font-size: 11px;
            color: rgba(255, 255, 255, 0.4);
        }

        .history-type {
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            padding: 2px 6px;
            border-radius: 6px;
            font-size: 10px;
            font-weight: 500;
        }

        /* Expand Button for Collapsed State */
        .expand-button {
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(59, 130, 246, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 0 8px 8px 0;
            padding: 12px 8px;
            color: #60a5fa;
            cursor: pointer;
            transition: all 0.3s ease;
            opacity: 0;
            pointer-events: none;
            z-index: 10;
        }

        .chat-history-panel.collapsed ~ .expand-button {
            opacity: 1;
            pointer-events: all;
        }

        .expand-button:hover {
            background: rgba(59, 130, 246, 0.25);
            transform: translateY(-50%) translateX(2px);
        }

        /* Main Chat Area */
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
            background: rgba(0, 0, 0, 0.3);
        }

        /* Chat Header */
        .chat-header {
            padding: 24px 32px;
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(25px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .chat-title {
            font-size: 20px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.95);
            letter-spacing: -0.02em;
        }

        .session-sources {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .attached-source {
            background: rgba(59, 130, 246, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 20px;
            padding: 6px 12px;
            font-size: 12px;
            color: #60a5fa;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
            opacity: 0;
            transform: scale(0.8) translateY(-10px);
            animation: slideIn 0.5s ease forwards;
        }

        @keyframes slideIn {
            to {
                opacity: 1;
                transform: scale(1) translateY(0);
            }
        }

        .source-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #3b82f6;
        }

        /* Chat Messages */
        .chat-messages {
            flex: 1;
            padding: 32px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .message {
            max-width: 80%;
            animation: messageSlide 0.5s ease;
        }

        @keyframes messageSlide {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.user {
            align-self: flex-end;
        }

        .message.assistant {
            align-self: flex-start;
        }

        .message-content {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 20px 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 15px;
            line-height: 1.6;
        }

        .message.user .message-content {
            background: rgba(59, 130, 246, 0.12);
            border-color: rgba(59, 130, 246, 0.3);
            color: rgba(255, 255, 255, 0.95);
        }

        .message-content pre {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 12px;
            overflow: auto;
        }

        .message-content code {
            background: rgba(0, 0, 0, 0.4);
            padding: 2px 4px;
            border-radius: 4px;
            font-family: Menlo, Monaco, monospace;
            font-size: 0.9em;
        }

        .message-content a {
            color: #60a5fa;
            text-decoration: underline;
        }

        /* Welcome State */
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            flex: 1;
            text-align: center;
            padding: 60px 40px;
        }

        .welcome-icon {
            width: 80px;
            height: 80px;
            border-radius: 20px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            margin-bottom: 24px;
            box-shadow: 0 20px 40px rgba(59, 130, 246, 0.25);
        }

        .welcome-title {
            font-size: 28px;
            font-weight: 700;
            color: rgba(255, 255, 255, 0.95);
            margin-bottom: 16px;
            letter-spacing: -0.02em;
        }

        .welcome-subtitle {
            font-size: 18px;
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 32px;
            max-width: 600px;
            line-height: 1.5;
        }

        .welcome-features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 24px;
            max-width: 800px;
            margin-bottom: 48px;
        }

        .feature-card {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(15px);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            text-align: left;
        }

        .feature-icon {
            font-size: 24px;
            margin-bottom: 12px;
        }

        .feature-title {
            font-size: 16px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 8px;
        }

        .feature-description {
            font-size: 14px;
            color: rgba(255, 255, 255, 0.6);
            line-height: 1.4;
        }

        .add-source-hint {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 24px 32px;
            border: 2px dashed rgba(59, 130, 246, 0.3);
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 16px;
            color: rgba(255, 255, 255, 0.7);
        }

        .add-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: rgba(59, 130, 246, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            color: #60a5fa;
        }

        /* Chat Input */
        .chat-input-container {
            padding: 24px 32px;
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(25px);
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            position: relative;
        }

        .input-wrapper {
            position: relative;
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }

        .input-wrapper:focus-within {
            border-color: rgba(59, 130, 246, 0.5);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .chat-input {
            width: 100%;
            background: transparent;
            border: none;
            padding: 48px 60px 48px 20px; /* Increased height by 3x */
            color: rgba(255, 255, 255, 0.9);
            font-size: 16px;
            resize: none;
            outline: none;
            font-family: inherit;
            min-height: 120px; /* Explicit minimum height */
        }

        .chat-input:disabled {
            color: rgba(255, 255, 255, 0.3);
            cursor: not-allowed;
        }

        .chat-input::placeholder {
            color: rgba(255, 255, 255, 0.4);
        }

        .send-button {
            position: absolute;
            right: 12px;
            bottom: 12px; /* Adjusted position for larger input */
            width: 48px;
            height: 48px;
            border-radius: 24px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            opacity: 0.5;
        }

        .send-button:enabled {
            opacity: 1;
        }

        .send-button:enabled:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
        }

        .input-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(25px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            font-size: 15px;
            color: rgba(255, 255, 255, 0.5);
            transition: all 0.3s ease;
            pointer-events: none;
        }

        .input-overlay.hidden {
            opacity: 0;
            pointer-events: none;
        }

        /* Drop Zone */
        .drop-zone {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(59, 130, 246, 0.08);
            backdrop-filter: blur(25px);
            border: 2px dashed rgba(59, 130, 246, 0.4);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            pointer-events: none;
            transition: all 0.3s ease;
            z-index: 100;
        }

        .drop-zone.active {
            opacity: 1;
            pointer-events: all;
        }

        .drop-message {
            text-align: center;
            color: #60a5fa;
            font-size: 18px;
            font-weight: 600;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.03);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(59, 130, 246, 0.3);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(59, 130, 246, 0.5);
        }

        /* Hidden state */
        .hidden {
            display: none !important;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .sidebar {
                width: 250px;
            }
            
            .chat-history-panel {
                width: 280px;
            }
            
            .chat-messages {
                padding: 16px;
            }
            
            .chat-header, .chat-input-container {
                padding: 16px 20px;
            }
        }
    </style>
</head>
<body>
    <!-- Animated Background Particles -->
    <div class="bg-particles" id="particles"></div>

    <!-- Main App Container -->
    <div class="app-container">
        <!-- Left Sidebar -->
        <div class="sidebar">
            <!-- Header with Org Selection -->
            <div class="sidebar-header">
                <div class="org-selector">
                    <div class="org-selector-content">
                        <div class="org-info">
                            <div class="org-avatar">UP</div>
                            <div class="org-details">
                                <h3>uspraveen's Org</h3>
                                <p>Professional</p>
                            </div>
                        </div>
                        <div class="dropdown-arrow">▼</div>
                    </div>
                </div>

                <div class="personal-selector">
                    <div class="personal-content">
                        <div class="personal-info">
                            <div class="personal-icon">👤</div>
                            <div class="org-details">
                                <h3>Personal</h3>
                                <p>Private workspace</p>
                            </div>
                        </div>
                        <div class="dropdown-arrow">▼</div>
                    </div>
                </div>
            </div>

            <!-- Navigation Tabs -->
            <div class="nav-tabs">
                <div class="nav-tab active">
                    <div class="nav-icon">💬</div>
                    <div class="nav-label">Ask</div>
                </div>
                <div class="nav-tab">
                    <div class="nav-icon">📚</div>
                    <div class="nav-label">Documentation</div>
                    <div class="nav-badge">Beta</div>
                </div>
                <div class="nav-tab">
                    <div class="nav-icon">⭐</div>
                    <div class="nav-label">Review</div>
                </div>
                <div class="nav-tab">
                    <div class="nav-icon">🔍</div>
                    <div class="nav-label">Search</div>
                </div>
            </div>

            <!-- Sources Section at Bottom -->
            <div class="sources-section">
                <div class="sources-header">
                    <svg class="github-logo" viewBox="0 0 24 24">
                        <path d="M12 0C5.374 0 0 5.373 0 12 0 17.302 3.438 21.8 8.207 23.387c.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/>
                    </svg>
                    <span class="sources-title">Sources</span>
                </div>

                <div class="sources-container">
                    <!-- Preview State -->
                    <div class="sources-preview">
                        <div class="preview-header">
                            <span class="preview-title">Connected Repositories</span>
                            <span class="repo-count">5</span>
                        </div>
                        <div class="preview-repos">
                            <div class="repo-avatar">DC</div>
                            <div class="repo-avatar">LF</div>
                            <div class="repo-avatar">NP</div>
                        </div>
                        <p class="expand-hint">Hover to expand and drag</p>
                    </div>

                    <!-- Expanded State -->
                    <div class="sources-expanded">
                        <div class="scroll-indicators">
                            <div class="scroll-arrow">▲</div>
                            <div class="scroll-arrow">▼</div>
                        </div>

                        <div class="repos-list">
                            <div class="repo-card" draggable="true" data-repo="uspraveen/dosu2">
                                <div class="repo-info">
                                    <div class="repo-icon">DC</div>
                                    <div class="repo-details">
                                        <h4>uspraveen/dosu-chat</h4>
                                        <p>AI-powered code intelligence</p>
                                    </div>
                                </div>
                            </div>

                            <div class="repo-card" draggable="true" data-repo="uspraveen/langChain">
                                <div class="repo-info">
                                    <div class="repo-icon">LF</div>
                                    <div class="repo-details">
                                        <h4>uspraveen/langChain</h4>
                                        <p>Visual flow builder for AI</p>
                                    </div>
                                </div>
                            </div>

                            <div class="repo-card" draggable="true" data-repo="uspraveen/nextjs-portfolio">
                                <div class="repo-info">
                                    <div class="repo-icon">NP</div>
                                    <div class="repo-details">
                                        <h4>uspraveen/nextjs-portfolio</h4>
                                        <p>Modern portfolio website</p>
                                    </div>
                                </div>
                            </div>

                            <div class="repo-card" draggable="true" data-repo="uspraveen/ai-agents">
                                <div class="repo-info">
                                    <div class="repo-icon">AA</div>
                                    <div class="repo-details">
                                        <h4>uspraveen/ai-agents</h4>
                                        <p>Autonomous AI agent framework</p>
                                    </div>
                                </div>
                            </div>

                            <div class="repo-card" draggable="true" data-repo="uspraveen/data-pipeline">
                                <div class="repo-info">
                                    <div class="repo-icon">DP</div>
                                    <div class="repo-details">
                                        <h4>uspraveen/data-pipeline</h4>
                                        <p>Real-time data processing</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Middle Panel - Chat History -->
        <div class="chat-history-panel" id="chatHistoryPanel">
            <div class="history-panel-header">
                <h2 class="history-panel-title">Ask</h2>
                <div class="panel-controls">
                    <button class="new-chat-btn">
                        <span>✏️</span>
                        New Chat
                    </button>
                    <button class="collapse-btn" id="collapseBtn">
                        <span>◀</span>
                    </button>
                </div>
            </div>

            <div class="search-box">
                <div class="search-icon">🔍</div>
                <input type="text" class="search-input" placeholder="Search...">
            </div>

            <div class="history-list">
                <div class="history-item active">
                    <div class="history-title-text">Understanding Chain creation process</div>
                    <div class="history-preview">how does chains work here</div>
                    <div class="history-meta">
                        <div class="history-time">Today</div>
                    </div>
                </div>
            

                

                <div class="history-item">
                    <div class="history-title-text">API rate limiting implementation</div>
                    <div class="history-preview">implement rate limiting for REST API endpoints</div>
                    <div class="history-meta">
                        <div class="history-time">Yesterday</div>
                    </div>
                </div>

                
            </div>
        </div>

        <!-- Expand Button (visible when collapsed) -->
        <button class="expand-button" id="expandBtn">
            <span>▶</span>
        </button>

        <!-- Main Chat Area -->
        <div class="chat-container">
            <!-- Drop Zone -->
            <div class="drop-zone" id="dropZone">
                <div class="drop-message">
                    <div>📁 Drop repository here to attach</div>
                    <div style="font-size: 14px; opacity: 0.7; margin-top: 8px;">Repository will be available for this conversation</div>
                </div>
            </div>

            <!-- Chat Header -->
            <div class="chat-header">
                <h1 class="chat-title">Understanding Neo4j graph generation process</h1>
                <div class="session-sources" id="sessionSources">
                    <!-- Attached sources will appear here -->
                </div>
            </div>

            <!-- Welcome State -->
            <div class="welcome-container" id="welcomeContainer">
                <div class="welcome-icon">🚀</div>
                <h2 class="welcome-title">Welcome to Dosu Intelligence</h2>
                <p class="welcome-subtitle">
                    Your fact-based search agent. To get started, attach a repository source and unlock intelligent code exploration, pattern discovery, and automated documentation.
                </p>

                <div class="welcome-features">
                    <div class="feature-card">
                        <div class="feature-icon">🔍</div>
                        <div class="feature-title">Smart Code Search</div>
                        <div class="feature-description">Find functions, classes, and patterns using natural language queries</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">📊</div>
                        <div class="feature-title">Codebase Analytics</div>
                        <div class="feature-description">Understand architecture, dependencies, and code relationships</div>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon">⚡</div>
                        <div class="feature-title">Instant Answers</div>
                        <div class="feature-description">Get explanations, examples, and guidance in real-time</div>
                    </div>
                </div>

                <div class="add-source-hint">
                    <div class="add-icon">+</div>
                    <div>
                        <strong>Ready to start?</strong> Drag a repository from the sources panel to begin your intelligent code exploration.
                    </div>
                </div>
            </div>

            <!-- Chat Messages (hidden initially) -->
            <div class="chat-messages hidden" id="chatMessages">
                <!-- Messages will appear here after source is attached -->
            </div>

            <!-- Chat Input -->
            <div class="chat-input-container">
                <div class="input-wrapper">
                    <textarea 
                        class="chat-input" 
                        placeholder="Ask anything about your code..."
                        rows="3"
                        id="chatInput"
                        disabled
                    ></textarea>
                    <button class="send-button" id="sendButton" disabled>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="m22 2-7 20-4-9-9-4 20-7z"/>
                        </svg>
                    </button>
                    <div class="input-overlay" id="inputOverlay">
                        <div class="add-icon">+</div>
                        Add a repository source to start chatting
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Application state
        let currentSessionId = 'session-' + Math.random().toString(36).substr(2, 9);
        let hasAttachedSources = false;
        let draggedRepo = null;
        let isProcessing = false;

        // DOM elements
        const welcomeContainer = document.getElementById('welcomeContainer');
        const chatMessages = document.getElementById('chatMessages');
        const chatInput = document.getElementById('chatInput');
        const sendButton = document.getElementById('sendButton');
        const inputOverlay = document.getElementById('inputOverlay');
        const sessionSources = document.getElementById('sessionSources');
        const dropZone = document.getElementById('dropZone');
        const chatHistoryPanel = document.getElementById('chatHistoryPanel');
        const collapseBtn = document.getElementById('collapseBtn');
        const expandBtn = document.getElementById('expandBtn');

        // Create animated background particles
        function createParticles() {
            const container = document.getElementById('particles');
            const particleCount = 60;
            
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.top = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 8 + 's';
                particle.style.animationDuration = (Math.random() * 4 + 4) + 's';
                container.appendChild(particle);
            }
        }

        // Panel collapse/expand functionality
        collapseBtn.addEventListener('click', () => {
            chatHistoryPanel.classList.add('collapsed');
        });

        expandBtn.addEventListener('click', () => {
            chatHistoryPanel.classList.remove('collapsed');
        });

        // Navigation functionality
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
            });
        });

        // History item selection
        document.querySelectorAll('.history-item').forEach(item => {
            item.addEventListener('click', () => {
                document.querySelectorAll('.history-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
            });
        });

        // Drag and Drop functionality - EXACT ORIGINAL
        document.querySelectorAll('.repo-card').forEach(card => {
            card.addEventListener('dragstart', (e) => {
                draggedRepo = e.target.dataset.repo;
                e.target.style.opacity = '0.5';
                dropZone.classList.add('active');
            });

            card.addEventListener('dragend', (e) => {
                e.target.style.opacity = '1';
                dropZone.classList.remove('active');
            });
        });

        // Drop zone handlers - EXACT ORIGINAL
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            if (draggedRepo) {
                attachRepository(draggedRepo);
                dropZone.classList.remove('active');
            }
        });

        function attachRepository(repoName) {
            // Check if already attached
            if (document.querySelector(`[data-attached="${repoName}"]`)) {
                return;
            }

            // Enable chat interface
            if (!hasAttachedSources) {
                enableChatInterface();
                hasAttachedSources = true;
            }

            const sourceElement = document.createElement('div');
            sourceElement.className = 'attached-source';
            sourceElement.dataset.attached = repoName;
            sourceElement.innerHTML = `
                <div class="source-dot"></div>
                ${repoName}
            `;

            sessionSources.appendChild(sourceElement);

            // Add success message
            addMessage('assistant', `✅ Repository "${repoName}" has been attached to this conversation. You can now ask questions about this codebase!`);
        }

        function enableChatInterface() {
            // Hide welcome container
            welcomeContainer.classList.add('hidden');
            
            // Show chat messages
            chatMessages.classList.remove('hidden');
            
            // Enable input
            chatInput.disabled = false;
            sendButton.disabled = false;
            inputOverlay.classList.add('hidden');
            
            // Add initial assistant message
            addMessage('assistant', `Great! I'm now connected to your repository:
                Go ahead and ask me anything! 🚀
            `);
        }

        function formatContent(content) {
            const container = document.createElement('div');
            container.innerHTML = marked.parse(content);

            // Highlight class and function names outside code blocks
            const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null);
            const classRegex = /\b[A-Z][A-Za-z0-9_]*\b/g;
            const funcRegex = /\b[a-zA-Z_][A-Za-z0-9_]*\(\)/g;
            const nodes = [];
            while (walker.nextNode()) {
                const node = walker.currentNode;
                const parentTag = node.parentNode.tagName;
                if (parentTag !== 'CODE' && parentTag !== 'PRE') {
                    nodes.push(node);
                }
            }
            nodes.forEach(n => {
                let html = n.nodeValue.replace(funcRegex, '<code>$&</code>').replace(classRegex, '<code>$&</code>');
                if (html !== n.nodeValue) {
                    const span = document.createElement('span');
                    span.innerHTML = html;
                    n.parentNode.replaceChild(span, n);
                }
            });

            return container.innerHTML;
        }

        function addMessage(type, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            const formatted = formatContent(content);
            messageDiv.innerHTML = `<div class="message-content">${formatted}</div>`;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Backend integration functions
        async function sendMessage() {
            const message = chatInput.value.trim();
            if (!message || !hasAttachedSources || isProcessing) return;

            isProcessing = true;
            addMessage('user', message);
            chatInput.value = '';

            // Reset textarea height
            chatInput.style.height = 'auto';

            // Add processing indicator
            const processingDiv = document.createElement('div');
            processingDiv.className = 'message assistant processing';
            processingDiv.innerHTML = '<div class="message-content">🔄 Enhancing your query...</div>';
            chatMessages.appendChild(processingDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;

            const phases = [
                '🔄 Enhancing your query...',
                '🛠️ Generating Cypher patterns...',
                '📦 Retrieving content...',
                '⚙️ Synthesizing answer...'
            ];
            let phaseIndex = 0;
            const phaseInterval = setInterval(() => {
                const text = phases[Math.min(phaseIndex, phases.length - 1)];
                processingDiv.querySelector('.message-content').innerText = text;
                phaseIndex++;
            }, 1200);


            try {
                // Send to backend API
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: message,
                        session_id: currentSessionId
                    })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                clearInterval(phaseInterval);

                const matches = result.debug_info?.results_count || 0;
                processingDiv.querySelector('.message-content').innerText = `📊 Found ${matches} matches`;
                await new Promise(r => setTimeout(r, 1200));
                processingDiv.remove();
                
                // Add response
                addMessage('assistant', result.response || result.assistant_message?.content || 'I processed your query successfully!');

            } catch (error) {
                console.error('Query failed:', error);
                clearInterval(phaseInterval);

                processingDiv.remove();
                addMessage('assistant', `Sorry, I encountered an error: ${error.message}. But I'm still here to help with your codebase questions!`);
            } finally {
                isProcessing = false;
            }
        }

        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Auto-resize textarea
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });

        // Initialize particles
        createParticles();
    </script>
</body>
</html>'''
    
    return HTMLResponse(content=html_content)

# API Routes
@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    """Process a query"""
    try:
        logger.info(f"Processing query for session {request.session_id}: {request.query[:100]}...")
        result = await bridge.process_query(request.session_id, request.query)
        logger.info(f"Query processed successfully for session {request.session_id}")
        return result
    except Exception as e:
        logger.error(f"Query endpoint failed: {e}")
        # Return a user-friendly error instead of raising HTTP exception
        return {
            "user_message": {"content": request.query, "type": "user"},
            "assistant_message": {
                "content": f"I'm having trouble connecting to the retriever service. Error: {str(e)}. Please try again or check if the retriever is running.",
                "type": "assistant"
            },
            "error": str(e),
            "session": {"id": request.session_id}
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "sessions_count": len(bridge.sessions),
        "retriever_url": bridge.retriever_url,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/sessions/{session_id}")
async def get_session_endpoint(session_id: str):
    """Get specific session - auto-creates if not found"""
    session = bridge.get_session(session_id)
    if not session:
        # Auto-create session
        session = ChatSession(
            id=session_id,
            title="New Chat",
            created_at=datetime.now().isoformat(),
            last_message_at=datetime.now().isoformat(),
            messages=[],
            attached_sources=[]
        )
        bridge.sessions[session_id] = session
        bridge._save_sessions()
        logger.info(f"Auto-created session via API: {session_id}")
    
    return {"session": session.model_dump()}

@app.get("/api/sessions")
async def get_sessions_endpoint():
    """Get all chat sessions"""
    return {"sessions": [session.model_dump() for session in bridge.sessions.values()]}

def main():
    """Main function to start the bridge server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dosu Intelligence Bridge Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--retriever-url", default="http://localhost:8001", 
                       help="URL of the retriever API (default: http://localhost:8001)")
    parser.add_argument("--data-dir", default="./data", 
                       help="Directory to store session data (default: ./data)")
    
    args = parser.parse_args()
    
    # Initialize bridge with custom settings
    global bridge
    bridge = BridgeServer(retriever_url=args.retriever_url, data_dir=args.data_dir)
    
    logger.info(f"🚀 Starting Dosu Intelligence Bridge Server")
    logger.info(f"📍 UI: http://{args.host}:{args.port}")
    logger.info(f"🔗 Retriever API: {args.retriever_url}")
    logger.info(f"💾 Data directory: {args.data_dir}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    main()