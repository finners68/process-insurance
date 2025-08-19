import os
import base64
import logging
import fitz  # PyMuPDF
import boto3
import uuid
from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_combined")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Required ENV vars
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
AWS_REGION = os.environ.get("AWS_REGION")
S3_BUCKET = os.environ.get("S3_BUCKET")
API_KEY = os.environ.get("API_KEY")

if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, S3_BUCKET, API_KEY]):
    raise EnvironmentError("Missing one or more required environment variables.")

# AWS clients
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

textract = boto3.client(
    "textract",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

# API key check
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

@app.post("/process-insurance-combined")
async def process_insurance_combined(request: Request, _: None = Depends(verify_api_key)):
    try:
        body = await request.json()
        if "file" not in body or "filename" not in body:
            return JSONResponse(status_code=400, content={"error": "Missing 'file' or 'filename'"})

        base64_file = body["file"]
        original_filename = body["filename"]

        # Decode base64 file
        try:
            file_bytes = base64.b64decode(base64_file)
            logger.info(f"📄 Decoded file size: {len(file_bytes)} bytes")
        except Exception:
            logger.error("❌ Failed to decode base64", exc_info=True)
            return JSONResponse(status_code=400, content={"error": "Invalid base64 file."})

        # Process PDF and upload each page as image to S3
        try:
            logger.info("🧼 Processing all PDF pages...")
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            image_keys = []
            filename_base = os.path.splitext(original_filename)[0]

            for page_number in range(len(doc)):
                page = doc.load_page(page_number)
                pix = page.get_pixmap(dpi=300)
                image_bytes = pix.tobytes("png")

                image_filename = f"{filename_base}-page{page_number + 1}-{uuid.uuid4()}.png"
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=image_filename,
                    Body=image_bytes,
                    ContentType="image/png"
                )
                image_keys.append(image_filename)
                logger.info(f"☁️ Uploaded page {page_number + 1} to S3 as {image_filename}")

        except Exception:
            logger.error("❌ Failed to flatten/upload PDF pages", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "PDF processing failed."})

        # Run Textract generic OCR (detect_document_text) on all images
        raw_text_pages = []
        try:
            logger.info("🧠 Running detect_document_text on all pages...")
            for key in image_keys:
                response = textract.detect_document_text(
                    Document={"S3Object": {"Bucket": S3_BUCKET, "Name": key}}
                )
                lines = [
                    block["Text"]
                    for block in response.get("Blocks", [])
                    if block["BlockType"] == "LINE"
                ]
                page_text = "\n".join(lines)
                raw_text_pages.append(page_text)

        except Exception:
            logger.warning("⚠️ Textract OCR failed", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "Textract OCR failed."})

        # Combine all pages' text
        raw_text = "\n\n".join(raw_text_pages)

        return JSONResponse(status_code=200, content={
            "raw_text": raw_text
        })

    except Exception:
        logger.error("🔥 Unexpected error", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
