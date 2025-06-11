from pinecone import Pinecone

pc = Pinecone(api_key="")
index = pc.Index("movies-walkthrough")

raw_data = "Movies about tragic love"
results = pc.inference.embed(
	model = "multilingual-e5-large",
	inputs=[raw_data],
	parameters={
		"input_type": "query",
		"truncate" : "END"
	}
)

print(results)

query_results = index.query(
	vector = results.data[0]['values'],
	top_k = 3,
	include_metadata=True
)

print(query_results)
