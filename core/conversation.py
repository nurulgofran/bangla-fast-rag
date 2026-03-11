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
        max_entries = MAX_HISTORY_TURNS * 2
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


# Words that are PURELY referential or adjectives — contain no core product nouns.
# If a query only contains these, it's considered a follow-up query and gets enriched.
STOP_AND_COMMON_WORDS = {
    # Question words (with variations & suffixes)
    "কি", "কী", "কত", "কতো", "কোন", "কোন্‌", "কোনটা", "কোনটার", "কেন", "ক্যানো", "কিভাবে", "কোথায়", "কবে", "কাকে", "কার",
    
    # Pronouns / reference words (with suffixes)
    "এটা", "এটার", "ইটা", "সেটা", "সেটার", "ওটা", "ওটার", "এইটা", "এইটার", "সেইটা", "সেইটার", "ওইটা", "ওইটার",
    "এর", "এদের", "তার", "তাদের", "ওদের", "এগুলো", "এগুলোর", "সেগুলো", "সেগুলোর", "ওগুলো", "ওগুলোর",
    "এই", "সেই", "ওই", "এখান", "সেখান", "এখানকার", "সেখানকার",
    
    # Common verbs / particles
    "আছে", "আছেন", "ছিল", "করে", "করেন", "করি", "করব", "করবো", "হয়", "হয়", "যায়", "যায়", "পাওয়া", "পাবো", "পাব", "দিবে", "দিবেন", "দেন", "দিন",
    "বলুন", "বলেন", "বলবেন", "বলবা", "জানান", "জানাবেন", "দেখান", "দেখাবেন", "দেখবো", "দেখব",
    "করুন", "চাই", "লাগবে", "নিতে", "কিনতে", "বিক্রি", "দিলে", "হলে",
    
    # Price / attribute / feature words (referential without a product noun)
    "দাম", "দামটা", "মূল্য", "টাকা", "টাকায়", "টাকার", "রং", "রঙ", "রংটা", "সাইজ", "সাইজটা", "ওজন", "রেট", "প্রাইস", "প্রাইসটা", "কালার",
    
    # Adjectives / Colors / Sizes (acting as referential qualifiers to a product)
    "সাদা", "কালো", "লাল", "নীল", "সবুজ", "হলুদ", "গোলাপি", "কমলা", "বাদামী", "বেগুনি", "সোনালী", "রুপালী", "কালোর", "সাদার",
    "বড়", "ছোট", "মাঝারি", "মিডিয়াম", "লং", "শর্ট", "লম্বা", "খাটো", "চিকন", "মোটা",
    "এক্সট্রা", "মিনি", "ডিলাক্স", "প্রিমিয়াম", "ক্লাসিক", "স্ট্যান্ডার্ড", "জাম্বো", "সুপার", "প্যাক", "প্লাটিনাম", "সিলভার", "গোল্ড", "ফ্যামিলি",
    
    # Conversational Fillers / Structure words / Polite markers
    "আর", "আরও", "অন্য", "বিস্তারিত", "ছবি", "ডেলিভারি", "অর্ডার",
    "আপনাদের", "আপনাগো", "কোম্পানি", "একটি", "একটা", "আমাকে", "আমারে", "আমি", "আপনি",
    "প্লিজ", "দয়া", "ধন্যবাদ", "তো", "না", "নাকি", "কিবা", "তাহলে", "ভাই", "ভাইয়া", "ভায়া",
    "আপন", "অবশ্যই", "আচ্ছা", "ঠিক", "খুব", "যে", "টা", "টি", "গুলো", "গুলি", "গুলা", "টায়",
    "একটু", "একটুখানি", "দিয়ে", "দিয়ে", "ওকে", "জি", "জী", "হ্যাঁ", "হুম", "বুঝলাম", "তাই", "দেখি", "মতন", "মতো", "রকম", "ভাবছি",
    "কিন্তু", "এবং", "ও"
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
    if not state.history or not state.current_topic:
        return query, False

    query_stripped = query.strip()
    import re
    clean_words = re.findall(r'[\u0980-\u09FF]+', query_stripped)
    query_words = set(clean_words)

    content_words = {w for w in query_words if w not in STOP_AND_COMMON_WORDS and len(w) > 1}

    if content_words:
        return query, False

    topic = state.current_query_topic or _extract_base_name(state.current_topic or "")
    if topic:
        enriched = f"{topic} {query_stripped}"
        return enriched, True

    return query, False


def _has_overlap(query: str, product_name: str) -> bool:
    """Check if query contains significant words from product name."""
    product_words = set(product_name.split())
    query_words = set(query.split())
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

    words = name.split()
    if len(words) >= 3:
        return " ".join(words[-2:])
    return name
