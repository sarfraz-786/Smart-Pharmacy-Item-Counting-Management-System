# ============================================================
# FASTAPI BACKEND
#
# Browser owns the webcam. This server only:
#   1. receives an uploaded frame at POST /detect
#   2. decodes it with OpenCV (in memory, no window)
#   3. hands it to detector.detect_objects()
#   4. returns JSON (counts, boxes, fps) to the browser
#
# It never opens cv2.VideoCapture() and never calls cv2.imshow().
# ============================================================

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from detector import detect_objects, model, CONFIDENCE


app = FastAPI(title="Pill & Capsule Counting API")


# ------------------------------------------------------------
# CORS
# The HTML page is served separately (e.g. Live Server / a
# local http.server) from a different origin than this API, so
# the browser needs explicit CORS permission to call /detect.
# Add any other local port you serve the HTML from.
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Yeh sabhi ports aur XAMPP local URLs ko baghair kisi restriction ke allow kar dega
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# HEALTH / INFO
# Open http://127.0.0.1:8000/ in a browser to confirm the
# server is up and to see the model's real class names — use
# these exact names as the keys in the HTML file's `products`
# object for billing.
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Pill & Capsule Counting API is running.",
        "classes": model.names,
        "default_confidence": CONFIDENCE,
    }


# ------------------------------------------------------------
# MAIN DETECTION ENDPOINT
# ------------------------------------------------------------

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "Empty file received — no image data was uploaded.",
                },
            )

        # Decode the uploaded bytes into an OpenCV BGR frame.
        # No cv2.VideoCapture, no window — this is purely an
        # in-memory image decode.
        file_bytes = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "Could not decode image. Send a valid JPEG/PNG frame.",
                },
            )

        results = detect_objects(frame)

        return {
            "success": True,
            "results": results,
        }

    except Exception as e:
        # Never fail silently — always return a readable error
        # so the browser console / UI can show what went wrong.
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )


# ------------------------------------------------------------
# RUN DIRECTLY (optional convenience — normally use:
#   uvicorn app:app --reload
# from the command line instead)
# ------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
