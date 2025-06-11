from pinecone import Pinecone

pc = Pinecone(api_key="")
index = pc.Index("llama-text-embed-v2-index")

raw_data = "Tell me the difference between \"Everyone\" and \"Everybody."
results = pc.inference.embed(
	model = "llama-text-embed-v2",
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
