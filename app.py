from fastapi import FastAPI
from routes.layout import layout_router

app = FastAPI()

app.include_router(layout_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(host='0.0.0.0', port=8080, app=app)