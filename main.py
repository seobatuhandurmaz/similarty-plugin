from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()

# CORS MIDDLEWARE: Tüm domainlere izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.batuhandurmaz.com"],  # Gerekirse ["https://www.batuhandurmaz.com"] yap
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ENV
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai = OpenAI(api_key=OPENAI_KEY)

@app.get("/")
def root():
    return {"status": "Semantic Linker API running"}

@app.get("/related/{slug}")
def get_related(slug: str, request: Request):
    domain = request.headers.get("origin", "default")
    domain = domain.replace("https://", "").replace("http://", "")
    
    sim_res = supabase.table("similarities")\
        .select("*")\
        .eq("source_slug", slug)\
        .eq("domain", domain)\
        .order("similarity_score", desc=True)\
        .limit(5)\
        .execute()

    return {"related": sim_res.data}

@app.post("/process")
def generate_embeddings_and_similarities():
    articles = supabase.table("articles").select("*").execute().data
    slugs = []
    vectors = []

    for article in articles:
        if article["embedding"] is None:
            res = openai.embeddings.create(
                input=article["content"],
                model="text-embedding-3-small"
            )
            embedding = res.data[0].embedding
            supabase.table("articles").update({
                "embedding": embedding
            }).eq("id", article["id"]).execute()

        slugs.append(article["slug"])
        vectors.append(article["embedding"])

    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            sim = float(cosine_similarity([vectors[i]], [vectors[j]])[0][0])
            supabase.table("similarities").insert({
                "source_slug": slugs[i],
                "target_slug": slugs[j],
                "domain": "batuhandurmaz.com",  # opsiyonel sabit domain
                "similarity_score": sim
            }).execute()

    return {"message": "Embedding ve benzerlik işlemi tamamlandı"}
