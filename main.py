from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import requests
from dotenv import load_dotenv

# === 1️⃣ Load environment variables ===
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")

print("DEBUG: SUPABASE_URL =", SUPABASE_URL)
print("DEBUG: SUPABASE_KEY =", "✅ Loaded" if SUPABASE_KEY else "❌ Missing")
print("DEBUG: RAPIDAPI_KEY =", "✅ Loaded" if RAPIDAPI_KEY else "❌ Missing")
print("DEBUG: ADZUNA_APP_ID =", "✅ Loaded" if ADZUNA_APP_ID else "❌ Missing")

# === 2️⃣ Initialize FastAPI + Supabase ===
app = FastAPI(title="INTERLINK AI Matcher", version="1.6.0")

# ✅ Allow frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] for Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


@app.get("/")
def root():
    """Health check endpoint"""
    return {"message": "✅ INTERLINK AI Matcher is running"}


@app.post("/match")
def match_internships(student_id: str = Query(..., description="UUID of the student")):
    """
    Fetch internships from Supabase + APIs (JSearch, Adzuna),
    and rank them using AI similarity with student profile.
    """

    # === 3️⃣ Fetch student profile ===
    try:
        student_res = supabase.table("students").select("*").eq("id", student_id).execute()
        if not student_res.data:
            return {"error": f"No student found with id {student_id}"}
        student = student_res.data[0]
    except Exception as e:
        return {"error": f"Database error fetching student: {str(e)}"}

    student_text = (
        " ".join(student.get("skills", [])) + " " + (student.get("bio") or "")
    ).lower()

    # === 4️⃣ Internal internships (Supabase) ===
    try:
        internal_res = supabase.table("internships").select("id,title,description,required_skills").execute()
        internal_data = internal_res.data or []
    except Exception as e:
        print("⚠️ Supabase fetch failed:", e)
        internal_data = []

    internal_df = pd.DataFrame(internal_data)
    if not internal_df.empty:
        internal_df["source"] = "supabase"
        internal_df["url"] = ""  # no external link
        internal_df["text"] = internal_df.apply(
            lambda x: (
                (x.get("title") or "") + " " +
                (x.get("description") or "") + " " +
                " ".join(x.get("required_skills", []))
            ).lower(),
            axis=1,
        )
    else:
        internal_df = pd.DataFrame(columns=["id", "title", "description", "text", "source", "url"])

    # === 5️⃣ JSearch (RapidAPI) ===
    jsearch_list = []
    try:
        if RAPIDAPI_KEY:
            skills_query = "+".join(student.get("skills", [])) or "internship"
            jsearch_url = f"https://jsearch.p.rapidapi.com/search?query={skills_query}+internship&page=1&num_pages=1"
            headers = {
                "x-rapidapi-key": RAPIDAPI_KEY,
                "x-rapidapi-host": "jsearch.p.rapidapi.com",
            }
            response = requests.get(jsearch_url, headers=headers, timeout=10)
            if response.status_code == 200:
                for job in response.json().get("data", [])[:10]:
                    jsearch_list.append({
                        "id": job.get("job_id", f"j_{len(jsearch_list)}"),
                        "title": job.get("job_title", "Untitled"),
                        "description": job.get("job_description", ""),
                        "url": job.get("job_apply_link", ""),  # ✅ actual job link
                        "required_skills": [],
                        "source": "external_jsearch"
                    })
                print(f"✅ Got {len(jsearch_list)} internships from JSearch")
            else:
                print(f"⚠️ JSearch returned {response.status_code}")
    except Exception as e:
        print("⚠️ JSearch API error:", e)

    # === 6️⃣ Adzuna ===
    adzuna_list = []
    try:
        if ADZUNA_APP_ID and ADZUNA_APP_KEY:
            adzuna_url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
            params = {
                "app_id": ADZUNA_APP_ID,
                "app_key": ADZUNA_APP_KEY,
                "results_per_page": 10,
                "what": "internship",
                "where": "India",
            }
            resp = requests.get(adzuna_url, params=params, timeout=10)
            if resp.status_code == 200:
                for job in resp.json().get("results", []):
                    adzuna_list.append({
                        "id": f"adz_{job.get('id', len(adzuna_list))}",
                        "title": job.get("title", "Untitled"),
                        "description": job.get("description", ""),
                        "url": job.get("redirect_url", ""),  # ✅ actual Adzuna link
                        "required_skills": [],
                        "source": "external_adzuna"
                    })
                print(f"✅ Got {len(adzuna_list)} internships from Adzuna")
            else:
                print(f"⚠️ Adzuna returned {resp.status_code}")
    except Exception as e:
        print("⚠️ Adzuna fetch failed:", e)

    # === 7️⃣ Combine all ===
    external_list = jsearch_list + adzuna_list
    external_df = pd.DataFrame(external_list)
    if not external_df.empty:
        external_df["text"] = external_df.apply(
            lambda x: ((x["title"] or "") + " " + (x["description"] or "")).lower(),
            axis=1,
        )

    all_internships = pd.concat([internal_df, external_df], ignore_index=True)
    all_internships.drop_duplicates(subset=["title", "description"], inplace=True)

    if all_internships.empty:
        return {"error": "No internships found (Supabase + APIs)"}

    # === 8️⃣ Compute AI similarity ===
    corpus = [student_text] + all_internships["text"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    all_internships["similarity"] = similarities

    # === 9️⃣ Sort & Return ===
    top_matches = all_internships.sort_values("similarity", ascending=False).head(10)
    results = top_matches[["id", "title", "description", "similarity", "source", "url"]].to_dict(orient="records")

    print(f"✅ Found {len(results)} matches for student {student_id}")
    return {
        "student_id": student_id,
        "matches_found": len(results),
        "matches": results
    }
