import os
import math
import logging
import threading
from typing import List, Dict, Any, Tuple
import datetime
from schemas.macro_schemas import ThemeCategory

log = logging.getLogger(__name__)

# NLP Lazy Loading Setup
_embedding_model = None
_embedding_model_lock = threading.Lock()
_ENABLE_NEWS_EMBEDDINGS = os.getenv("ENABLE_NEWS_EMBEDDINGS", "false").lower() == "true"

def _get_embedding_model():
    global _embedding_model
    if not _ENABLE_NEWS_EMBEDDINGS:
        return None

    if _embedding_model is None:
        with _embedding_model_lock:
            # Double-checked locking
            if _embedding_model is None:
                start_time = datetime.datetime.now()
                log.info("Loading SentenceTransformer model...")
                try:
                    from sentence_transformers import SentenceTransformer
                    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    elapsed = (datetime.datetime.now() - start_time).total_seconds()
                    log.info(f"Loaded SentenceTransformer in {elapsed:.2f}s")
                except Exception as e:
                    log.error(f"Failed to load SentenceTransformer: {e}")
                    _embedding_model = "FAILED"
                    
    if _embedding_model == "FAILED":
        return None
    return _embedding_model


def calculate_freshness(age_hours: int, category: ThemeCategory) -> Tuple[float, str]:
    """
    Calculate freshness score and reason based on category-aware decay curves.
    """
    if age_hours < 0:
        age_hours = 0
        
    if category in [ThemeCategory.RISK_SENTIMENT, ThemeCategory.EARNINGS]:
        # Fast Exponential Decay (Half-life ~ 24h, fully stale at 48h)
        if age_hours <= 12:
            return 1.0, f"Fresh {category.value} (<=12h)"
        elif age_hours <= 24:
            return 0.75, f"Recent {category.value} (12-24h)"
        elif age_hours <= 48:
            return 0.25, f"Fading {category.value} (24-48h)"
        else:
            return 0.05, f"Stale {category.value} (>48h)"
            
    elif category in [ThemeCategory.POLICY, ThemeCategory.LIQUIDITY, ThemeCategory.GEOPOLITICS]:
        # Slow Linear Decay (Half-life ~ 15-30 days)
        if age_hours <= 168: # 7 days
            return 1.0, f"Current {category.value} (<=7d)"
        elif age_hours <= 720: # 30 days
            decay = max(0.2, 1.0 - ((age_hours - 168) / (720 - 168)) * 0.8)
            return round(decay, 2), f"Maturing {category.value} (7-30d)"
        else:
            return 0.1, f"Old {category.value} (>30d)"
            
    else:
        # Default medium decay (Growth, Inflation)
        if age_hours <= 72: # 3 days
            return 1.0, f"Recent {category.value} (<=3d)"
        elif age_hours <= 336: # 14 days
            decay = max(0.3, 1.0 - ((age_hours - 72) / (336 - 72)) * 0.7)
            return round(decay, 2), f"Maturing {category.value} (3-14d)"
        else:
            return 0.1, f"Old {category.value} (>14d)"


def calculate_event_confidence(sources_count: int) -> float:
    """
    Calculate event confidence using log-scale saturation.
    3-5 tier-1 sources max out the confidence score.
    """
    if sources_count <= 0:
        return 0.0
    # Cap at ~5 sources
    cap = 5.0
    val = math.log1p(sources_count) / math.log1p(cap)
    return round(min(1.0, val), 2)


def canonicalize_source(source_name: str) -> str:
    """Normalize source names for grouping and priority."""
    name = source_name.lower().strip()
    if "bloomberg" in name:
        return "Bloomberg"
    if "reuters" in name:
        return "Reuters"
    if "wsj" in name or "wall street journal" in name:
        return "WSJ"
    if "cnbc" in name:
        return "CNBC"
    if "ft" in name or "financial times" in name:
        return "Financial Times"
    if "yahoo finance" in name:
        return "Yahoo Finance"
    if "investing.com" in name:
        return "Investing.com"
    return source_name.strip()


def _get_source_tier(canonical_name: str) -> int:
    """Return tier (1 = Highest)."""
    tier_1 = {"Bloomberg", "Reuters", "WSJ", "Financial Times"}
    tier_2 = {"CNBC", "Yahoo Finance", "Investing.com"}
    if canonical_name in tier_1:
        return 1
    if canonical_name in tier_2:
        return 2
    return 3


def _jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def group_similar_news(news_items: List[Dict[str, Any]], threshold: float = 0.75) -> List[List[Dict[str, Any]]]:
    """
    Group similar news items to deduplicate and prevent echo chamber.
    Fallback to Jaccard similarity if embedding model is not available.
    Expects news_items to have keys: 'title', 'source', 'published_at'
    """
    if not news_items:
        return []

    model = _get_embedding_model()
    embeddings = None
    
    if model is not None:
        try:
            from sentence_transformers import util
            titles = [item.get('title', '') for item in news_items]
            embeddings = model.encode(titles, convert_to_tensor=True)
        except Exception as e:
            log.error(f"Error during embedding generation: {e}")
            embeddings = None

    clusters = []
    used_indices = set()
    
    for i in range(len(news_items)):
        if i in used_indices:
            continue
            
        current_cluster = [news_items[i]]
        used_indices.add(i)
        
        for j in range(i + 1, len(news_items)):
            if j in used_indices:
                continue
                
            sim_score = 0.0
            if embeddings is not None:
                # Use cosine similarity
                try:
                    sim_score = util.cos_sim(embeddings[i], embeddings[j]).item()
                except Exception:
                    sim_score = _jaccard_similarity(news_items[i].get('title', ''), news_items[j].get('title', ''))
            else:
                # Fallback to Jaccard
                # Lower threshold for Jaccard since it's stricter
                sim_score = _jaccard_similarity(news_items[i].get('title', ''), news_items[j].get('title', ''))
                
            eff_threshold = threshold if embeddings is not None else max(0.4, threshold - 0.3)
            
            if sim_score >= eff_threshold:
                current_cluster.append(news_items[j])
                used_indices.add(j)
                
        clusters.append(current_cluster)
        
    return clusters


def select_representative_news(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select the best representative title based on tier and recency.
    """
    if not cluster:
        return {}
        
    if len(cluster) == 1:
        rep = cluster[0].copy()
        rep['sources_count'] = 1
        return rep
        
    # Sort by (tier ASC, published_at DESC)
    def sort_key(item):
        canon = canonicalize_source(item.get('source', ''))
        tier = _get_source_tier(canon)
        # Handle naive or missing dates by assigning a very old date for sorting
        pub = item.get('published_at')
        if not pub:
            pub = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        elif pub.tzinfo is None:
             pub = pub.replace(tzinfo=datetime.timezone.utc)
        return (tier, -pub.timestamp())
        
    sorted_cluster = sorted(cluster, key=sort_key)
    rep = sorted_cluster[0].copy()
    rep['sources_count'] = len(cluster)
    return rep
