"""
Validate vector database format
"""

import os
import pickle

print("🔍 Validating Vector Database")
print("=" * 50)

# Load database
try:
    with open("vector_db.pkl", 'rb') as f:
        db = pickle.load(f)
    print("✅ Database loaded successfully")
except Exception as e:
    print(f"❌ Error loading database: {e}")
    exit(1)

# Check required keys
required_keys = ['chunks', 'embeddings', 'metadata', 'created_at']
for key in required_keys:
    if key not in db:
        print(f"❌ Missing key: {key}")
    else:
        print(f"✅ {key}: Found")

# Check data consistency
chunks = db.get('chunks', [])
embeddings = db.get('embeddings', [])
metadata = db.get('metadata', [])

print(f"\n📊 Data Summary:")
print(f"   Chunks: {len(chunks)}")
print(f"   Embeddings: {len(embeddings)}")
print(f"   Metadata: {len(metadata)}")

# Check first chunk
if chunks:
    print(f"\n📝 First chunk preview:")
    print(f"   Length: {len(chunks[0])} characters")
    print(f"   Preview: {chunks[0][:100]}...")

# Check first embedding
if embeddings:
    import numpy as np
    first_embedding = np.array(embeddings[0])
    print(f"\n🧠 First embedding:")
    print(f"   Dimensions: {first_embedding.shape}")
    print(f"   All zeros: {np.all(first_embedding == 0)}")

# Save a clean version
print(f"\n💾 Creating clean version...")
clean_db = {
    'chunks': chunks,
    'embeddings': embeddings,
    'metadata': metadata,
    'created_at': db.get('created_at'),
    'total_chunks': len(chunks)
}

with open("vector_db_clean.pkl", 'wb') as f:
    pickle.dump(clean_db, f)

print("✅ Clean database saved as vector_db_clean.pkl")