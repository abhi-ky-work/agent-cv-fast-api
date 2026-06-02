from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from app.agent.graph import agent_graph

app = FastAPI(
    title="AgentCV API",
    version="1.0",
    description="A simple API server for AgentCV using Langserve",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
async def redirect_root_to_docs():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/docs")

add_routes(app, agent_graph, path="/agent")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
