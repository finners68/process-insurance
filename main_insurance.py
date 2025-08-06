import os
import io
import json
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

# CORS (for dev/testing)
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

        # Flatten PDF to image
        try:
            logger.info("🧼 Flattening PDF...")
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=300)
            image_bytes = pix.tobytes("png")
        except Exception:
            logger.error("❌ Failed to flatten PDF", exc_info=True)
            return JSONResponse(status_code=400, content={"error": "PDF flattening failed."})

        # Upload to S3
        filename_base = os.path.splitext(original_filename)[0]
        filename = f"{filename_base}-{uuid.uuid4()}.png"

        try:
            logger.info(f"☁️ Uploading to S3: {filename}")
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=filename,
                Body=image_bytes,
                ContentType="image/png"
            )
        except Exception:
            logger.error("❌ S3 upload failed", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "S3 upload failed."})

        # Textract: analyze_document (FORMS)
        structured_fields = {}
        try:
            logger.info("🧠 Calling analyze_document...")
            response = textract.analyze_document(
                Document={"S3Object": {"Bucket": S3_BUCKET, "Name": filename}},
                FeatureTypes=["FORMS"]
            )
            blocks = response.get("Blocks", [])
            block_map = {b["Id"]: b for b in blocks}

            for block in blocks:
                if block["BlockType"] == "KEY_VALUE_SET" and "KEY" in block.get("EntityTypes", []):
                    key = ""
                    val = ""
                    for rel in block.get("Relationships", []):
                        if rel["Type"] == "CHILD":
                            key = " ".join([
                                block_map[child_id]["Text"]
                                for child_id in rel["Ids"]
                                if block_map[child_id]["BlockType"] == "WORD"
                            ])
                        if rel["Type"] == "VALUE":
                            for value_id in rel["Ids"]:
                                value_block = block_map.get(value_id)
                                if not value_block or "Relationships" not in value_block:
                                    continue
                                for child_rel in value_block["Relationships"]:
                                    if child_rel["Type"] == "CHILD":
                                        val = " ".join([
                                            block_map[child_id]["Text"]
                                            for child_id in child_rel["Ids"]
                                            if block_map[child_id]["BlockType"] == "WORD"
                                        ])
                    if key and val:
                        structured_fields[key.strip()] = val.strip()
        except Exception:
            logger.warning("⚠️ analyze_document failed", exc_info=True)

        # Textract: detect_document_text
        raw_text = ""
        try:
            logger.info("📃 Calling detect_document_text...")
            response = textract.detect_document_text(
                Document={"S3Object": {"Bucket": S3_BUCKET, "Name": filename}}
            )
            lines = [
                block["Text"]
                for block in response.get("Blocks", [])
                if block["BlockType"] == "LINE"
            ]
            raw_text = "\n".join(lines)
        except Exception:
            logger.warning("⚠️ detect_document_text failed", exc_info=True)

        return JSONResponse(status_code=200, content={
            "structured_fields": structured_fields,
            "raw_text": raw_text
        })

    except Exception:
        logger.error("🔥 Unexpected error", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
