"""
Debug vector database loading
"""

import os
import pickle

print("🔍 Debugging Vector Database Loading")
print("=" * 50)

# Check if file exists
db_path = "textbook_vector_db.pkl"
print(f"\n📁 Checking file: {db_path}")
print(f"   Exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    print(f"   Size: {os.path.getsize(db_path):,} bytes")
    print(f"   Size: {os.path.getsize(db_path) / 1024 / 1024:.2f} MB")

    # Try to load it
    print("\n📖 Attempting to load...")
    try:
        with open(db_path, 'rb') as f:
            db_data = pickle.load(f)

        print("✅ Successfully loaded!")

        # Check contents
        print(f"\n📊 Database contents:")
        print(f"   - Chunks: {len(db_data.get('chunks', []))}")
        print(f"   - Embeddings: {len(db_data.get('embeddings', []))}")
        print(f"   - Metadata: {len(db_data.get('metadata', []))}")
        print(f"   - Keys in database: {list(db_data.keys())}")

        # Check sample data
        if db_data.get('chunks'):
            print(f"\n📝 Sample chunk preview:")
            print(f"   First chunk length: {len(db_data['chunks'][0])}")
            print(f"   Preview: {db_data['chunks'][0][:100]}...")

    except Exception as e:
        print(f"❌ Error loading database: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ File does not exist!")