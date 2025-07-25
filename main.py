from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os

# FastAPI uygulaması oluşturuluyor
app = FastAPI()

# CORS middleware tanımı (her yerden erişim için '*' kullanıldı)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # istersen sadece domainini yaz: ["https://www.batuhandurmaz.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ortam değişkenleri
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Supabase ve OpenAI istemcilerini başlat
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai = OpenAI(api_key=OPENAI_KEY)

# Root endpoint
@app.get("/")
def root():
    return {"status": "Semantic Linker API running"}

# Embedding + benzerlik hesaplama endpoint'i
@app.post("/process")
def generate_embeddings_and_similarities(request: Request):
    # Domain parametresi al
    domain = request.query_params.get("domain", "default")

    # Bu domaine ait içerikleri al
    articles = supabase.table("articles").select("*")\
        .eq("domain", domain).execute().data

    slugs = []
    vectors = []

    # Embedding üret ve Supabase'e yaz
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

    # Benzerlikleri hesapla ve Supabase'e yaz
    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            sim = float(cosine_similarity([vectors[i]], [vectors[j]])[0][0])
            supabase.table("similarities").insert({
                "source_slug": slugs[i],
                "target_slug": slugs[j],
                "similarity_score": sim,
                "domain": domain
            }).execute()

    return {"message": f"{domain} için işlem tamamlandı"}

# Semantic iç link önerileri dönen endpoint
@app.get("/related/{slug}")
def get_related_articles(slug: str, request: Request):
    # Domain parametresi al
    domain = request.query_params.get("domain", "default")

    # similarities tablosundan en yakın içerikleri çek
    sim_res = supabase.table("similarities")\
        .select("*")\
        .eq("source_slug", slug)\
        .eq("domain", domain)\
        .order("similarity_score", desc=True)\
        .limit(5)\
        .execute()

    related = []

    # her benzer içerik için article başlığı çek
    for item in sim_res.data:
        target = supabase.table("articles")\
            .select("title, slug")\
            .eq("slug", item["target_slug"])\
            .eq("domain", domain)\
            .limit(1)\
            .execute()

        if target.data:
            related.append({
                "title": target.data[0]["title"],
                "slug": target.data[0]["slug"],
                "score": item["similarity_score"]
            })

    return JSONResponse(content={"related": related})
