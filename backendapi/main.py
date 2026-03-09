from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/search")
async def search(
    text: str = Form(""),
    method: str = Form(...),
    image: UploadFile | None = File(None)
):
    return {
        "results": [
            {
                "id": 1,
                "title": "Demo Product",
                "price": 99.99,
                "rating": 4.5,
                "image_url": "https://placehold.co/400",
                "category": "demo",
                "final_score": 0.92,
                "clip_score": 0.8,
                "text_score": 0.9
            }
        ],
        "diagnostics": {},
        "total": 1
    }
@app.get("/")
def root():
    return {"message": "Nexus Search API running"}