"""
FAISS index management for fast face matching
"""
import numpy as np
import faiss
from scipy.spatial.distance import cosine
from config import SIMILARITY_THRESHOLD, DUPLICATE_THRESHOLD

# Global variables
faiss_index = None
person_id_mapping = []

def build_faiss_index(known_faces):
    global faiss_index, person_id_mapping
    
    embeddings_list = []
    person_id_mapping = []
    
    for person_id, embeddings in known_faces.items():
        for embedding in embeddings:
            try:
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                embedding = embedding.astype('float32')
                
                if embedding.ndim == 1 and len(embedding) > 0:
                    embeddings_list.append(embedding)
                    person_id_mapping.append(person_id)
            except Exception:
                continue
    
    if embeddings_list:
        try:
            embeddings_array = np.array(embeddings_list)
            embedding_dim = embeddings_array.shape[1]
            
            faiss_index = faiss.IndexFlatIP(embedding_dim)
            faiss.normalize_L2(embeddings_array)
            faiss_index.add(embeddings_array)
            
            print(f"🚀 FAISS index built: {len(embeddings_list)} embeddings from {len(known_faces)} people")
        except Exception as e:
            print(f"❌ FAISS error: {e}")
            faiss_index = None
            person_id_mapping = []

def find_matching_person_fast(face_embedding, similarity_threshold=SIMILARITY_THRESHOLD):
    global faiss_index, person_id_mapping
    
    if faiss_index is None or len(person_id_mapping) == 0:
        return None, 0
    
    try:
        query = face_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query)
        similarities, indices = faiss_index.search(query, min(5, faiss_index.ntotal))
        
        if len(similarities[0]) > 0 and similarities[0][0] > similarity_threshold:
            best_idx = indices[0][0]
            if 0 <= best_idx < len(person_id_mapping):
                return person_id_mapping[best_idx], float(similarities[0][0])
        
        return None, 0
    except Exception:
        return None, 0

def check_for_duplicate_during_registration(face_embedding, known_faces, threshold=DUPLICATE_THRESHOLD):
    global faiss_index, person_id_mapping
    
    # FAISS search
    if faiss_index is not None and len(person_id_mapping) > 0:
        try:
            query = face_embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query)
            similarities, indices = faiss_index.search(query, min(3, faiss_index.ntotal))
            
            for similarity, idx in zip(similarities[0], indices[0]):
                if similarity > threshold and 0 <= idx < len(person_id_mapping):
                    matched_id = person_id_mapping[idx]
                    print(f"🔍 DUPLICATE DETECTED: {matched_id} (similarity: {similarity:.4f})")
                    return True, matched_id, float(similarity)
        except Exception:
            pass
    
    # Manual fallback
    best_match_id, best_similarity = None, 0
    for person_id, embeddings in known_faces.items():
        for stored_embedding in embeddings:
            try:
                similarity = 1 - cosine(face_embedding, np.array(stored_embedding))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = person_id
            except Exception:
                continue
    
    if best_similarity > threshold:
        print(f"🔍 DUPLICATE DETECTED: {best_match_id} (similarity: {best_similarity:.4f})")
        return True, best_match_id, best_similarity
    
    return False, None, best_similarity