# pip install "pinecone[grpc]"
from pinecone import Pinecone
pc = Pinecone(api_key="")

# To get the unique host for an index, 
# see https://docs.pinecone.io/guides/data/target-an-index
index = pc.Index(host="https://llama-text-embed-v2-index-8sgaw5u.svc.aped-4627-b74a.pinecone.io")

index.delete(delete_all=True)
