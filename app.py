from fastapi import FastAPI
from routes.predict_route import predict_router

app = FastAPI()

app.include_router(predict_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(host="0.0.0.0", port=8080, app=app)
