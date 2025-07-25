from fastapi import FastAPI
from fastapi.responses import JSONResponse
from supabase import create_client
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai = OpenAI(api_key=OPENAI_KEY)

@app.get("/")
def root():
    return {"status": "Semantic Linker API running"}

@app.post("/process")
def generate_embeddings_and_similarities():
    # 1. İçerikleri çek
    articles = supabase.table("articles").select("*").execute().data

    slugs = []
    vectors = []

    # 2. Embedding üret ve Supabase'e yaz
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

    # 3. Benzerlik hesapla ve similarities tablosuna yaz
    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            sim = float(cosine_similarity([vectors[i]], [vectors[j]])[0][0])
            supabase.table("similarities").insert({
                "source_slug": slugs[i],
                "target_slug": slugs[j],
                "similarity_score": sim
            }).execute()

    return {"message": "Embedding ve benzerlik işlemi tamamlandı"}

@app.get("/related/{slug}")
def get_related_articles(slug: str):
    # similarity tablosundan ilgili slug'a ait eşleşmeleri çek
    sim_res = supabase.table("similarities")\
        .select("*")\
        .eq("source_slug", slug)\
        .order("similarity_score", desc=True)\
        .limit(5)\
        .execute()
    
    related = []

    for item in sim_res.data:
        # Hedef slug'a karşılık gelen article'ı çekiyoruz
        target = supabase.table("articles")\
            .select("title, slug")\
            .eq("slug", item["target_slug"])\
            .limit(1)\
            .execute()

        if target.data:
            related.append({
                "title": target.data[0]["title"],
                "slug": target.data[0]["slug"],
                "score": item["similarity_score"]
            })

    return JSONResponse(content={"related": related})
