#!/usr/bin/env python3
"""
Vertex AI Audio Batch Processing script with failover and strict JSON schema output.
Compliant with 2025/2026 google-genai SDK standards.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Any

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VertexBatchProcessor")

# Ensure required libraries are importable
try:
    from google import genai
    from google.genai.types import CreateBatchJobConfig
    from google.api_core import exceptions
except ImportError:
    logger.critical(
        "Required libraries are missing. Please install them using:\n"
        "pip install google-genai google-api-core google-cloud-storage"
    )
    sys.exit(1)

# Ranked Priority List for Compliant Multimodal Models
MODEL_PRIORITIES = [
    "gemini-3.6-flash",       # Priority 1: High speed, top efficiency
    "gemini-3.6-pro",         # Priority 2: High intelligence
    "gemini-2.5-flash",       # Priority 3: Speed, cost efficiency, bulk data
    "gemini-2.5-pro",         # Priority 4: Advanced reasoning (fallback for noise/overlap)
    "gemini-1.5-flash-002"    # Priority 3: Stable previous generation production model
]

# Valid Server Regions for Batch GenAI to bypass quota limits
SERVER_REGIONS = [
    "us-central1",            # Iowa: Highest quotas, lowest latency
    "europe-west1",           # Belgium: Primary European fallback
    "asia-east1"              # Taiwan: Primary Asian fallback
]

# Strict Response Schema for Vocal Transcript metadata extraction
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "transcript": {
            "type": "STRING",
            "description": "The exact transcribed spoken text from the audio."
        },
        "english_translation": {
            "type": "STRING",
            "description": "The translation of the transcribed text into English."
        },
        "gender": {
            "type": "STRING",
            "description": "Voice audio analysis identifying the speaker gender: 'male', 'female', or 'unknown'."
        },
        "age_group": {
            "type": "STRING",
            "description": "Voice audio analysis identifying the speaker age group: 'young', 'adult', 'old', or 'unknown'."
        }
    },
    "required": ["transcript", "english_translation", "gender", "age_group"]
}

def create_input_jsonl_locally(audio_uris: List[str], mime_type: str = "audio/mp3") -> str:
    """
    Generates JSONL string locally where each line maps to a GenerateContentRequest
    with the strict output schema constraint.
    """
    lines = []
    prompt = (
        "Analyze the provided audio file. You must return a structured JSON response containing: "
        "1. 'transcript': The original spoken text. "
        "2. 'english_translation': The translation of the text into English. "
        "3. 'gender': Audio analysis identifying the speaker as male, female, or unknown. "
        "4. 'age_group': Audio analysis identifying the speaker as young, adult, old, or unknown."
    )
    
    for gcs_uri in audio_uris:
        request_obj = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "file_data": {
                                    "file_uri": gcs_uri,
                                    "mime_type": mime_type
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": RESPONSE_SCHEMA
                }
            }
        }
        lines.append(json.dumps(request_obj, ensure_ascii=False))
        
    return "\n".join(lines)

def submit_batch_job_with_failover(
    project_id: str,
    input_gcs_uri: str,
    output_gcs_uri: str
) -> Any:
    """
    Attempts to submit a batch job using a matrix of model priorities and regional endpoints
    to gracefully bypass quota or Server Busy limits.
    """
    logger.info("Initializing multi-model, multi-region failover loop...")
    
    for model in MODEL_PRIORITIES:
        for region in SERVER_REGIONS:
            logger.info(f"Attempting Batch Job creation. Model: {model} | Region: {region}")
            
            try:
                # Client initialization standard using the unified google-genai SDK
                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=region
                )
                
                # Create batch prediction standard
                config = CreateBatchJobConfig(dest=output_gcs_uri)
                job = client.batches.create(
                    model=model,
                    src=input_gcs_uri,
                    config=config
                )
                
                logger.info(
                    f"SUCCESS: Batch job created successfully!\n"
                    f"Job Resource Name: {job.name}\n"
                    f"Model: {model}\n"
                    f"Region: {region}"
                )
                return job
                
            except (exceptions.ResourceExhausted, exceptions.InternalServerError) as limit_err:
                logger.warning(
                    f"FAILOVER: Model '{model}' in Region '{region}' encountered limit/server error.\n"
                    f"Exception details: {limit_err}\n"
                    f"Retrying next failover combination in 2 seconds..."
                )
                time.sleep(2)
            except Exception as other_err:
                logger.error(
                    f"FAILOVER: Unexpected error with Model '{model}' in Region '{region}'.\n"
                    f"Exception details: {other_err}\n"
                    f"Retrying next failover combination in 2 seconds..."
                )
                time.sleep(2)
                
    raise RuntimeError(
        "FATAL: Failover matrix exhausted. Could not submit batch job to any model/region combination."
    )

def main():
    parser = argparse.ArgumentParser(
        description="Vertex AI Multimodal Batch Audio Processing Client with automatic failover."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Google Cloud Platform Project ID."
    )
    parser.add_argument(
        "--input-uri",
        required=True,
        help="GCS URI of the input JSONL file (e.g. gs://my-bucket/inputs.jsonl)."
    )
    parser.add_argument(
        "--output-uri",
        required=True,
        help="GCS URI prefix folder for output results (e.g. gs://my-bucket/outputs/)."
    )
    parser.add_argument(
        "--generate-local-jsonl",
        help="Local file path to write generated JSONL request lines to (optional)."
    )
    parser.add_argument(
        "--audio-uris-file",
        help="Path to a text file containing GCS audio file URIs (one per line) to help generate input JSONL."
    )
    parser.add_argument(
        "--mime-type",
        default="audio/mp3",
        help="Mime type of audio files when generating local JSONL (default: audio/mp3)."
    )

    args = parser.parse_args()

    # Generate JSONL if file input list is provided
    if args.audio_uris_file:
        if not args.generate_local_jsonl:
            logger.error("--generate-local-jsonl path must be provided when utilizing --audio-uris-file.")
            sys.exit(1)
            
        logger.info(f"Reading GCS Audio URIs from: {args.audio_uris_file}")
        try:
            with open(args.audio_uris_file, "r", encoding="utf-8") as f:
                uris = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Found {len(uris)} audio files. Generating JSONL requests...")
            jsonl_content = create_input_jsonl_locally(uris, args.mime_type)
            
            with open(args.generate_local_jsonl, "w", encoding="utf-8") as out_f:
                out_f.write(jsonl_content)
            logger.info(f"Successfully generated local JSONL request file at: {args.generate_local_jsonl}")
            logger.info("Upload this file to GCS first, and then run this script with --input-uri targeting it.")
            
        except Exception as file_err:
            logger.error(f"Failed to read file or generate JSONL: {file_err}")
            sys.exit(1)

    # Submit Batch Job
    try:
        submit_batch_job_with_failover(
            project_id=args.project,
            input_gcs_uri=args.input_uri,
            output_gcs_uri=args.output_uri
        )
    except Exception as run_err:
        logger.critical(f"Batch prediction job submission failed: {run_err}")
        sys.exit(1)

if __name__ == "__main__":
    main()
