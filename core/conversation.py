"""
Conversation state management & coreference resolution.
Tracks entities from retrieval results and enriches follow-up queries.
"""
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAX_HISTORY_TURNS


@dataclass
class ConversationState:
    """Maintains conversation context for coreference resolution."""
    history: list[dict] = field(default_factory=list)
    current_topic: str | None = None
    current_query_topic: str | None = None  # User's original search terms
    last_retrieved_products: list[dict] = field(default_factory=list)

    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn."""
        self.history.append({"role": role, "content": content})
        # Trim to max history
        max_entries = MAX_HISTORY_TURNS * 2  # user + assistant per turn
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def update_topic_from_results(self, results: list[dict], user_query: str = "") -> None:
        """
        Extract and store the current topic from retrieval results and user query.
        Called after retrieval — NOT in the Q2 hot path.
        """
        self.last_retrieved_products = results
        if results:
            self.current_topic = results[0].get("name_bn", None)

        # Also store the user's search terms as topic
        if user_query:
            import re
            clean_words = re.findall(r'[\u0980-\u09FF]+', user_query)
            content = [w for w in clean_words if w not in STOP_AND_COMMON_WORDS and len(w) > 1]
            if content:
                self.current_query_topic = " ".join(content)

    def get_history_text(self) -> str:
        """Format conversation history for LLM prompts."""
        lines = []
        for entry in self.history:
            role_label = "ব্যবহারকারী" if entry["role"] == "user" else "সহকারী"
            lines.append(f"{role_label}: {entry['content']}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear conversation state."""
        self.history.clear()
        self.current_topic = None
        self.current_query_topic = None
        self.last_retrieved_products.clear()


# Words that are PURELY referential — contain no product information
# If a query consists ONLY of these words, it's a follow-up
STOP_AND_COMMON_WORDS = {
    # Question words
    "কি", "কী", "কত", "কোন", "কেন", "কিভাবে", "কোথায়",
    # Pronouns / reference words
    "এটা", "সেটা", "ওটা", "এর", "তার", "এগুলো", "সেগুলো", "ওগুলো",
    "এই", "সেই", "ওই", "এখান", "সেখান",
    # Common verbs / particles
    "আছে", "করে", "হয়", "যায়", "পাওয়া", "দিন", "বলুন", "জানান", "দেখান",
    "করুন", "চাই", "লাগবে", "নিতে", "কিনতে", "বিক্রি",
    # Price / attribute words (referential without a product noun)
    "দাম", "মূল্য", "টাকা", "রং", "সাইজ", "ওজন",
    # Filler / structure words
    "আর", "আরও", "অন্য", "বিস্তারিত", "ছবি", "ডেলিভারি", "অর্ডার",
    "আপনাদের", "কোম্পানি", "একটি", "একটা", "আমাকে", "আমি", "আপনি",
    "প্লিজ", "দয়া", "ধন্যবাদ",
}


def enrich_query(query: str, state: ConversationState) -> tuple[str, bool]:
    """
    Enrich a follow-up query with context from conversation state.

    Logic:
    1. No history → no enrichment
    2. Query has content words (potential product names) → NEW topic, no enrichment
    3. Query is purely referential (only stop words) → follow-up, enrich with topic

    Returns: (enriched_query, was_enriched)
    """
    # No history = first query, no enrichment needed
    if not state.history or not state.current_topic:
        return query, False

    query_stripped = query.strip()
    # Strip punctuation from words for matching
    import re
    clean_words = re.findall(r'[\u0980-\u09FF]+', query_stripped)  # Extract only Bangla chars
    query_words = set(clean_words)

    # Remove stop/common words to find content words
    content_words = {w for w in query_words if w not in STOP_AND_COMMON_WORDS and len(w) > 1}

    # If query has content words → user is asking about a NEW topic
    # e.g., "আপনাদের কোম্পানি কি দই বিক্রি করে?" → content_words = {"দই"}
    if content_words:
        return query, False

    # Query is purely referential (no content words)
    # e.g., "দাম কত টাকা?" → content_words = {} → it's a follow-up
    # Prefer user's query topic ("ইলেকট্রনিক্স পণ্য") over product name ("পাওয়ার ব্যাংক")
    topic = state.current_query_topic or _extract_base_name(state.current_topic or "")
    if topic:
        enriched = f"{topic} {query_stripped}"
        return enriched, True

    return query, False


def _has_overlap(query: str, product_name: str) -> bool:
    """Check if query contains significant words from product name."""
    product_words = set(product_name.split())
    query_words = set(query.split())
    # Remove very short common words
    product_words = {w for w in product_words if len(w) > 2}
    overlap = product_words & query_words
    return len(overlap) >= 1


def _extract_base_name(full_name: str) -> str:
    """
    Extract the base product name, removing brand prefixes and quality tags.
    E.g., 'ইস্পাহানি পাওয়ার ব্যাংক' -> 'পাওয়ার ব্যাংক'
    """
    if not full_name:
        return ""
    import re
    name = re.sub(r'\(.*?\)', '', full_name).strip()
    name = re.sub(r'-\s*\S+$', '', name).strip()

    # Keep last 2 meaningful words (product type), drop the first (brand)
    words = name.split()
    if len(words) >= 3:
        return " ".join(words[-2:])
    return name
