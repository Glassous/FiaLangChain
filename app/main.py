import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.agent import router as agent_router
from app.core.config import settings

app = FastAPI(
    title="FiaLangChain Microservice",
    description="Independent Python LangChain & LangGraph agent service",
    version="1.0.0"
)

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent"])

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "FiaLangChain",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
