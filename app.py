import os
import base64
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize OpenAI client using the environment variable
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # Ensure the environment variable is set

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create the folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load tests_data.json containing the pre-defined test embeddings
with open('tests_data.json', 'r') as f:
    test_data = json.load(f)

def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to calculate cosine similarity between two vectors
def calculate_cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

@app.post("/analyze")
async def analyze_prescription(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Save the uploaded file to the local folder
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Read the image file and encode it as base64
        with open(filepath, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Use the OpenAI client to get a response about the image
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please extract the test names from this prescription image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                        },
                    },
                ],
            }],
            max_tokens=300,
        )

        # Extract test names from the response content
        test_names = []
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content
            
            # Check for the introduction line and extract names that follow
            lines = content.splitlines()
            for line in lines:
                line = line.strip()
                # Look for lines that start with a number indicating test names
                if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
                    test_name = line.split(".", 1)[1].strip()  # Extract after the number and period
                    test_names.append(test_name)

        # Clean up the uploaded file
        os.remove(filepath)

        # For each test name, generate embeddings using OpenAI Embedding API
        results = []
        for test_name in test_names:
            embedding_response = client.embeddings.create(
                input=test_name,
                model="text-embedding-ada-002"  # Using the latest embedding model
            )

            # Access embedding vector properly using dot notation
            embedding_vector = embedding_response.data[0].embedding

            # Find the most similar test from the JSON data
            most_similar_test = None
            highest_similarity = -1

            # Compare with embeddings from tests_data.json
            for test in test_data:
                similarity = calculate_cosine_similarity(embedding_vector, test['test_embedding'])
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar_test = test
            
            # Append test name, its embedding, and the most similar test from tests_data.json
            results.append({
                "most_similar_test": {
                    "test_name": most_similar_test['test_name'],
                    "test_code": most_similar_test['test_code'],
                    "similarity_score": highest_similarity
                }
            })

        # Print results for debugging purposes
        print(results)

        # Return test names with their corresponding embeddings and the most similar test
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
