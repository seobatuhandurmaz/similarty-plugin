from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()

# Tüm domainlerden gelen isteklere CORS izni ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm domainlere izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    # Domain'i gelen isteğin başlığından al
    domain = request.headers.get("origin", "default")

    # similarities tablosundan en benzerleri çek
    sim_res = supabase.table("similarities")\
        .select("*")\
        .eq("source_slug", slug)\
        .eq("domain", domain)\
        .order("similarity_score", desc=True)\
        .limit(5)\
        .execute()

    results = []
    for sim in sim_res.data:
        article_res = supabase.table("articles")\
            .select("title, slug")\
            .eq("slug", sim["target_slug"])\
            .eq("domain", domain)\
            .limit(1)\
            .execute()
        if article_res.data:
            results.append(article_res.data[0])

    return results

@app.post("/process")
def generate_embeddings_and_similarities():
    articles = supabase.table("articles").select("*").execute().data

    slugs = []
    vectors = []
    domains = []

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
        domains.append(article.get("domain", "default"))

    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            if domains[i] == domains[j]:
                sim = float(cosine_similarity([vectors[i]], [vectors[j]])[0][0])
                supabase.table("similarities").insert({
                    "source_slug": slugs[i],
                    "target_slug": slugs[j],
                    "similarity_score": sim,
                    "domain": domains[i]
                }).execute()

    return {"message": "Embedding ve benzerlik işlemi tamamlandı"}
