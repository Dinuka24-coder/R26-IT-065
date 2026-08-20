_existing_collections = set()


async def refresh_collection_cache(db):
    """Cache which collections exist. Called once at startup."""
    global _existing_collections
    try:
        names = await db.list_collection_names()
        _existing_collections = set(names)
        print(f"📋 Collections cached: {sorted(_existing_collections)}")
    except Exception as e:
        print(f"⚠️ Could not cache collections: {e}")
        _existing_collections = {"users", "patients", "pneumothorax_results"}


def collection_exists(name: str) -> bool:
    return name in _existing_collections


def add_collection(name: str):
    """Call this after creating a new collection at runtime."""
    _existing_collections.add(name)