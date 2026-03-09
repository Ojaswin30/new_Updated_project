from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import sys

# ------------------------------------------------
# Make ML module importable
# ------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT_DIR))

# Pipeline imports
from ml.src.pipeline.run_late_fusion import run_late_fusion
from ml.src.pipeline.run_early_fusion import run_early_fusion

# ------------------------------------------------
# FastAPI app
# ------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# Health check
# ------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------
# Root endpoint
# ------------------------------------------------

@app.get("/")
def root():
    return {"message": "Nexus Search API running"}


# ------------------------------------------------
# Search API
# ------------------------------------------------

@app.post("/api/search")
async def search(
    text: str = Form(""),
    method: str = Form(...),
    image: UploadFile | None = File(None)
):

    image_path = None

    # Save uploaded image temporarily
    if image:
        upload_dir = ROOT_DIR / "tmp_uploads"
        upload_dir.mkdir(exist_ok=True)

        image_path = upload_dir / image.filename

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

    # ------------------------------------------------
    # Execute ML pipeline
    # ------------------------------------------------

    try:

        if method == "late_fusion":
            results = run_late_fusion(
                text=text,
                image_path=image_path
            )

        elif method == "early_fusion":
            results = run_early_fusion(
                text=text,
                image_path=image_path
            )

        else:
            return {"error": f"Unknown search method: {method}"}

    except Exception as e:
        return {"error": str(e)}

    # ------------------------------------------------
    # Convert results to API format
    # ------------------------------------------------

    formatted_results = []

    for r in results.get("results", [])[:20]:
        fields = r.get("fields", {})

        formatted_results.append({
            "id": r.get("product_id"),
            "title": r.get("title"),
            "price": fields.get("price"),
            "rating": fields.get("rating"),
            "image_url": fields.get("image_url"),
            "category": fields.get("category"),
            "final_score": r.get("final_score"),
            "clip_score": r.get("visual_score"),
            "text_score": r.get("constraint_score")
        })

    return {
        "results": formatted_results,
        "diagnostics": {
            "query": text,
            "method": method
        },
        "total": len(formatted_results)
    }