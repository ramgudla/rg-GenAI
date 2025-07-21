from sentence_transformers import SentenceTransformer

sentences = ["Apple is a fruit", "Car is a vehicle"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = model.encode(sentences)

#print(len(embeddings[0]))

#print(embeddings)


from gensim.models import Word2Vec

sentences = [
"Couchbase is a distributed NoSQL database.",
"Couchbase Capella provides flexibility and scalability.",
"Couchbase supports SQL++ for querying JSON documents.",
"Couchbase Mobile extends the database to the edge.",
"Couchbase has a built-in Full Text Search Engine"
]

# Preprocess the sentences: tokenize and lower case
processed_sentences = [sentence.lower().split() for sentence in sentences]

# Train the Word2Vec model
model = Word2Vec(sentences=processed_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get the vector for a word
word_vector = model.wv['couchbase']

# Print the vector
print(word_vector)
