# websocket_manager.py - Real-time WebSocket management
import asyncio
import json
import time
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, asdict
import uuid

from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis

logger = logging.getLogger(__name__)

@dataclass
class AnalysisEvent:
    id: str
    type: str
    timestamp: datetime
    data: Dict[str, Any]
    risk_score: int
    token_address: str

@dataclass
class SystemMetrics:
    timestamp: datetime
    total_analyses: int
    high_risk_count: int
    avg_risk_score: float
    analyses_per_hour: int
    response_time_ms: float
    error_rate: float
    cache_hit_rate: float
    active_connections: int

class MetricsCollector:
    def __init__(self):
        self.analysis_history: deque = deque(maxlen=1000)
        self.response_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
    
    def record_analysis(self, event: AnalysisEvent):
        self.analysis_history.append(event)
        self.total_requests += 1
    
    def record_response_time(self, time_ms: float):
        self.response_times.append(time_ms)
    
    def record_error(self):
        self.error_count += 1
        self.total_requests += 1
    
    def record_cache_hit(self):
        self.cache_hits += 1
    
    def record_cache_miss(self):
        self.cache_misses += 1
    
    def get_current_metrics(self, active_connections: int) -> SystemMetrics:
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)
        
        # Filter recent analyses
        recent_analyses = [
            event for event in self.analysis_history
            if event.timestamp > one_hour_ago
        ]
        
        # Calculate metrics
        total_analyses = len(self.analysis_history)
        high_risk_count = len([e for e in self.analysis_history if e.risk_score >= 60])
        avg_risk_score = (
            sum(e.risk_score for e in self.analysis_history) / total_analyses
            if total_analyses > 0 else 0
        )
        analyses_per_hour = len(recent_analyses)
        
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0
        )
        
        error_rate = (
            (self.error_count / self.total_requests * 100)
            if self.total_requests > 0 else 0
        )
        
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            (self.cache_hits / total_cache_requests * 100)
            if total_cache_requests > 0 else 0
        )
        
        return SystemMetrics(
            timestamp=now,
            total_analyses=total_analyses,
            high_risk_count=high_risk_count,
            avg_risk_score=round(avg_risk_score, 1),
            analyses_per_hour=analyses_per_hour,
            response_time_ms=round(avg_response_time, 1),
            error_rate=round(error_rate, 2),
            cache_hit_rate=round(cache_hit_rate, 1),
            active_connections=active_connections
        )

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_info: Dict[WebSocket, Dict] = {}
        self.subscriptions: Dict[str, Set[WebSocket]] = defaultdict(set)
    
    async def connect(self, websocket: WebSocket, connection_id: str = None):
        await websocket.accept()
        self.active_connections.add(websocket)
        
        connection_id = connection_id or str(uuid.uuid4())
        self.connection_info[websocket] = {
            "id": connection_id,
            "connected_at": datetime.utcnow(),
            "subscriptions": set()
        }
        
        logger.info(f"WebSocket connected: {connection_id}")
        return connection_id
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
            # Remove from subscriptions
            connection_info = self.connection_info.get(websocket, {})
            for subscription in connection_info.get("subscriptions", set()):
                self.subscriptions[subscription].discard(websocket)
            
            del self.connection_info[websocket]
            logger.info(f"WebSocket disconnected: {connection_info.get('id', 'unknown')}")
    
    def subscribe(self, websocket: WebSocket, topic: str):
        if websocket in self.active_connections:
            self.subscriptions[topic].add(websocket)
            self.connection_info[websocket]["subscriptions"].add(topic)
    
    def unsubscribe(self, websocket: WebSocket, topic: str):
        self.subscriptions[topic].discard(websocket)
        if websocket in self.connection_info:
            self.connection_info[websocket]["subscriptions"].discard(topic)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        if websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Failed to send message to websocket: {e}")
                self.disconnect(websocket)
    
    async def broadcast(self, message: dict, topic: str = None):
        if topic:
            connections = self.subscriptions.get(topic, set())
        else:
            connections = self.active_connections.copy()
        
        if connections:
            disconnected = []
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message, default=str))
                except Exception as e:
                    logger.error(f"Failed to broadcast to websocket: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for connection in disconnected:
                self.disconnect(connection)
    
    def get_connection_count(self) -> int:
        return len(self.active_connections)
    
    def get_connection_stats(self) -> Dict:
        return {
            "total_connections": len(self.active_connections),
            "subscriptions": {
                topic: len(connections) 
                for topic, connections in self.subscriptions.items()
            },
            "connections": [
                {
                    "id": info["id"],
                    "connected_at": info["connected_at"],
                    "subscriptions": list(info["subscriptions"])
                }
                for info in self.connection_info.values()
            ]
        }

class RealTimeManager:
    def __init__(self, redis_url: str = None):
        self.connection_manager = ConnectionManager()
        self.metrics_collector = MetricsCollector()
        self.redis_client = None
        self.redis_url = redis_url
        self.event_queue = asyncio.Queue()
        self.alert_thresholds = {
            "high_risk_rate": 0.3,  # 30% high risk analyses
            "error_rate": 0.05,     # 5% error rate
            "response_time": 5000,  # 5 second response time
        }
        
    async def initialize(self):
        """Initialize Redis connection and start background tasks"""
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("Redis connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Start background tasks
        asyncio.create_task(self.process_events())
        asyncio.create_task(self.broadcast_metrics())
        asyncio.create_task(self.check_alerts())
    
    async def connect_websocket(self, websocket: WebSocket) -> str:
        """Handle new WebSocket connection"""
        connection_id = await self.connection_manager.connect(websocket)
        
        # Send initial data
        await self.send_initial_data(websocket)
        
        return connection_id
    
    async def disconnect_websocket(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        self.connection_manager.disconnect(websocket)
    
    async def send_initial_data(self, websocket: WebSocket):
        """Send initial dashboard data to new connection"""
        try:
            metrics = self.metrics_collector.get_current_metrics(
                self.connection_manager.get_connection_count()
            )
            
            # Send welcome message
            await self.connection_manager.send_personal_message({
                "type": "welcome",
                "payload": {
                    "connection_id": self.connection_manager.connection_info[websocket]["id"],
                    "server_time": datetime.utcnow().isoformat(),
                    "metrics": asdict(metrics)
                }
            }, websocket)
            
            # Send recent analyses
            recent_analyses = list(self.metrics_collector.analysis_history)[-10:]
            for event in recent_analyses:
                await self.connection_manager.send_personal_message({
                    "type": "new_analysis",
                    "payload": asdict(event)
                }, websocket)
            
        except Exception as e:
            logger.error(f"Failed to send initial data: {e}")
    
    async def handle_websocket_message(self, websocket: WebSocket, message: dict):
        """Handle incoming WebSocket messages"""
        try:
            msg_type = message.get("type")
            payload = message.get("payload", {})
            
            if msg_type == "subscribe":
                topic = payload.get("topic")
                if topic:
                    self.connection_manager.subscribe(websocket, topic)
                    await self.connection_manager.send_personal_message({
                        "type": "subscribed",
                        "payload": {"topic": topic}
                    }, websocket)
            
            elif msg_type == "unsubscribe":
                topic = payload.get("topic")
                if topic:
                    self.connection_manager.unsubscribe(websocket, topic)
                    await self.connection_manager.send_personal_message({
                        "type": "unsubscribed",
                        "payload": {"topic": topic}
                    }, websocket)
            
            elif msg_type == "ping":
                await self.connection_manager.send_personal_message({
                    "type": "pong",
                    "payload": {"timestamp": datetime.utcnow().isoformat()}
                }, websocket)
            
            elif msg_type == "request_metrics":
                metrics = self.metrics_collector.get_current_metrics(
                    self.connection_manager.get_connection_count()
                )
                await self.connection_manager.send_personal_message({
                    "type": "metrics_update",
                    "payload": asdict(metrics)
                }, websocket)
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def record_analysis(self, analysis_data: dict):
        """Record a new analysis and broadcast to connected clients"""
        try:
            # Create analysis event
            event = AnalysisEvent(
                id=str(uuid.uuid4()),
                type="token_analysis",
                timestamp=datetime.utcnow(),
                data=analysis_data,
                risk_score=analysis_data.get("risk_score", 0),
                token_address=analysis_data.get("token_info", {}).get("address", "")
            )
            
            # Record metrics
            self.metrics_collector.record_analysis(event)
            
            # Queue event for processing
            await self.event_queue.put(event)
            
            # Cache in Redis if available
            if self.redis_client:
                try:
                    await self.redis_client.lpush(
                        "recent_analyses", 
                        json.dumps(asdict(event), default=str)
                    )
                    await self.redis_client.ltrim("recent_analyses", 0, 99)  # Keep last 100
                except Exception as e:
                    logger.error(f"Redis caching failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to record analysis: {e}")
    
    async def record_response_time(self, time_ms: float):
        """Record response time for metrics"""
        self.metrics_collector.record_response_time(time_ms)
    
    async def record_error(self):
        """Record an error for metrics"""
        self.metrics_collector.record_error()
    
    async def record_cache_hit(self):
        """Record cache hit for metrics"""
        self.metrics_collector.record_cache_hit()
    
    async def record_cache_miss(self):
        """Record cache miss for metrics"""
        self.metrics_collector.record_cache_miss()
    
    async def process_events(self):
        """Background task to process events"""
        while True:
            try:
                event = await self.event_queue.get()
                
                # Broadcast to all connected clients
                await self.connection_manager.broadcast({
                    "type": "new_analysis",
                    "payload": {
                        "token": {
                            "symbol": event.data.get("token_info", {}).get("symbol", "UNK"),
                            "address": event.token_address
                        },
                        "risk_score": event.risk_score,
                        "timestamp": event.timestamp.isoformat(),
                        "confidence": event.data.get("confidence_level", 0)
                    }
                })
                
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                await asyncio.sleep(1)
    
    async def broadcast_metrics(self):
        """Background task to broadcast metrics periodically"""
        while True:
            try:
                await asyncio.sleep(30)  # Broadcast every 30 seconds
                
                metrics = self.metrics_collector.get_current_metrics(
                    self.connection_manager.get_connection_count()
                )
                
                await self.connection_manager.broadcast({
                    "type": "metrics_update",
                    "payload": asdict(metrics)
                })
                
            except Exception as e:
                logger.error(f"Error broadcasting metrics: {e}")
    
    async def check_alerts(self):
        """Background task to check for alerts"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                metrics = self.metrics_collector.get_current_metrics(
                    self.connection_manager.get_connection_count()
                )
                
                alerts = []
                
                # Check high risk rate
                if metrics.total_analyses > 10:  # Only check if we have enough data
                    high_risk_rate = metrics.high_risk_count / metrics.total_analyses
                    if high_risk_rate > self.alert_thresholds["high_risk_rate"]:
                        alerts.append({
                            "type": "high_risk_rate",
                            "message": f"High risk token rate is {high_risk_rate:.1%} (threshold: {self.alert_thresholds['high_risk_rate']:.1%})",
                            "severity": "warning"
                        })
                
                # Check error rate
                if metrics.error_rate > self.alert_thresholds["error_rate"] * 100:
                    alerts.append({
                        "type": "high_error_rate",
                        "message": f"Error rate is {metrics.error_rate:.1f}% (threshold: {self.alert_thresholds['error_rate'] * 100:.1f}%)",
                        "severity": "error"
                    })
                
                # Check response time
                if metrics.response_time_ms > self.alert_thresholds["response_time"]:
                    alerts.append({
                        "type": "slow_response",
                        "message": f"Average response time is {metrics.response_time_ms:.0f}ms (threshold: {self.alert_thresholds['response_time']}ms)",
                        "severity": "warning"
                    })
                
                # Broadcast alerts
                for alert in alerts:
                    await self.connection_manager.broadcast({
                        "type": "alert",
                        "payload": alert
                    })
                
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        metrics = self.metrics_collector.get_current_metrics(
            self.connection_manager.get_connection_count()
        )
        
        return {
            "metrics": asdict(metrics),
            "connections": self.connection_manager.get_connection_stats(),
            "uptime": time.time() - self.metrics_collector.start_time,
            "redis_connected": self.redis_client is not None,
            "queue_size": self.event_queue.qsize()
        }

# Enhanced main.py with WebSocket support
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import time

# Initialize real-time manager
real_time_manager = RealTimeManager(os.getenv("REDIS_URL"))

# Enhanced FastAPI app
app = FastAPI(title="Ultimate Rug Checker API", version="2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize real-time manager on startup"""
    await real_time_manager.initialize()
    logger.info("Real-time manager initialized")

# WebSocket endpoint
@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    connection_id = await real_time_manager.connect_websocket(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await real_time_manager.handle_websocket_message(websocket, message)
            
    except WebSocketDisconnect:
        await real_time_manager.disconnect_websocket(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await real_time_manager.disconnect_websocket(websocket)

# Enhanced rugcheck endpoint with real-time support
@app.post("/rugcheck")
async def rugcheck_endpoint(request: RugCheckRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    
    try:
        # Validate token address
        token_address = request.token_address.strip()
        if not token_address or len(token_address) < 32:
            await real_time_manager.record_error()
            raise HTTPException(status_code=400, detail="Invalid token address")
        
        # Perform analysis
        result = await rug_checker.analyze_token_ultimate(token_address)
        
        # Record response time
        response_time = (time.time() - start_time) * 1000
        await real_time_manager.record_response_time(response_time)
        
        if not result.get("success"):
            await real_time_manager.record_error()
            raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))
        
        # Record analysis in real-time system
        background_tasks.add_task(real_time_manager.record_analysis, result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        await real_time_manager.record_error()
        logger.error(f"Endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Dashboard stats endpoint
@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    return real_time_manager.get_stats()

# Health check endpoint with metrics
@app.get("/health")
async def health_check():
    stats = real_time_manager.get_stats()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": stats["uptime"],
        "active_connections": stats["connections"]["total_connections"],
        "total_analyses": stats["metrics"]["total_analyses"],
        "redis_connected": stats["redis_connected"]
    }

# Advanced caching decorator
from functools import wraps
import hashlib
import pickle

def cache_result(ttl: int = 300, key_prefix: str = ""):
    """Advanced caching decorator with Redis support"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key_data = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            if real_time_manager.redis_client:
                try:
                    cached_data = await real_time_manager.redis_client.get(cache_key)
                    if cached_data:
                        await real_time_manager.record_cache_hit()
                        return pickle.loads(cached_data)
                except Exception as e:
                    logger.debug(f"Cache retrieval failed: {e}")
            
            # Record cache miss
            await real_time_manager.record_cache_miss()
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if real_time_manager.redis_client and result:
                try:
                    serialized = pickle.dumps(result)
                    await real_time_manager.redis_client.setex(cache_key, ttl, serialized)
                except Exception as e:
                    logger.debug(f"Cache storage failed: {e}")
            
            return result
        return wrapper
    return decorator

# Rate limiting with Redis
from collections import defaultdict
from time import time as current_time

class RateLimiter:
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.local_cache = defaultdict(list)
    
    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed under rate limit"""
        now = current_time()
        
        if self.redis_client:
            try:
                # Use Redis for distributed rate limiting
                pipe = self.redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, now - window)
                pipe.zcard(key)
                pipe.zadd(key, {str(now): now})
                pipe.expire(key, window)
                results = await pipe.execute()
                
                current_requests = results[1]
                return current_requests < limit
                
            except Exception as e:
                logger.error(f"Redis rate limiting failed: {e}")
                # Fall back to local rate limiting
        
        # Local rate limiting
        requests = self.local_cache[key]
        
        # Remove old requests
        self.local_cache[key] = [req_time for req_time in requests if req_time > now - window]
        
        if len(self.local_cache[key]) < limit:
            self.local_cache[key].append(now)
            return True
        
        return False

# Initialize rate limiter
rate_limiter = RateLimiter(real_time_manager.redis_client)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    client_ip = request.client.host
    
    # Apply rate limiting to rugcheck endpoint
    if request.url.path == "/rugcheck":
        allowed = await rate_limiter.is_allowed(f"rate_limit:{client_ip}", 10, 60)  # 10 requests per minute
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Please try again later."}
            )
    
    response = await call_next(request)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )