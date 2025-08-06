import os
import io
import json
import base64
import logging
import fitz  # PyMuPDF
import boto3
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS Environment Variables
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
AWS_REGION = os.environ.get("AWS_REGION")
S3_BUCKET = os.environ.get("S3_BUCKET")

if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, S3_BUCKET]):
    raise EnvironmentError("Missing one or more AWS environment variables.")

# AWS Clients
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

@app.post("/process-insurance")
async def process_insurance(request: Request):
    try:
        logger.info("📥 Received request to /process-insurance")

        body = await request.json()
        if "file" not in body or "filename" not in body:
            return JSONResponse(status_code=400, content={"error": "Missing 'file' or 'filename' in request body."})

        base64_file = body["file"]
        original_filename = body["filename"]

        logger.info(f"📦 Base64 input starts: {base64_file[:30]}...")

        try:
            file_bytes = base64.b64decode(base64_file)
            logger.info(f"📄 Decoded file size: {len(file_bytes)} bytes")
        except Exception:
            logger.error("❌ Failed to decode base64", exc_info=True)
            return JSONResponse(status_code=400, content={"error": "Invalid base64 string."})

        # Flatten PDF
        try:
            logger.info("🧼 Flattening PDF using PyMuPDF...")
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=300)
            image_bytes = pix.tobytes("png")
            logger.info(f"📄 Flattened image size: {len(image_bytes)} bytes")
        except Exception:
            logger.error("❌ PDF flattening failed", exc_info=True)
            return JSONResponse(status_code=400, content={"error": "Could not flatten PDF."})

        # Create S3 filename
        filename_base = os.path.splitext(original_filename)[0]
        filename = f"{filename_base}-{uuid.uuid4()}.png"

        # Upload to S3
        try:
            logger.info(f"☁️ Uploading image to S3 as: {filename}")
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=filename,
                Body=image_bytes,
                ContentType="image/png",
            )
        except Exception:
            logger.error("❌ Failed to upload to S3", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "S3 upload failed."})

        # Call Textract AnalyzeDocument with FORMS and TABLES
        try:
            logger.info("🧠 Calling Textract analyze_document (FORMS, TABLES)...")
            response = textract.analyze_document(
                Document={"S3Object": {"Bucket": S3_BUCKET, "Name": filename}},
                FeatureTypes=["FORMS", "TABLES"]
            )
        except Exception:
            logger.error("❌ Textract analyze_document failed", exc_info=True)
            return JSONResponse(status_code=400, content={"error": "Document analysis failed."})

        # Extract key-value pairs
        blocks = response.get("Blocks", [])
        block_map = {b["Id"]: b for b in blocks}
        key_values = {}

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
                    key_values[key.strip()] = val.strip()

        return JSONResponse(status_code=200, content={"fields": key_values})

    except Exception:
        logger.error("🔥 Unhandled exception during /process-insurance", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error."})
