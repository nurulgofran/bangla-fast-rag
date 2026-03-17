"""
Dual response strategy — template-based (fast) + LLM-based (rich).
Template responses run in <1ms for the hot path.
LLM responses used for conversational/complex queries.
"""
from core.llm import groq_llm


# ─── Template-Based Response (HOT PATH — <1ms) ───

PRICE_QUERY_KEYWORDS = ["দাম", "কত", "মূল্য", "প্রাইস", "টাকা", "খরচ"]
AVAILABILITY_KEYWORDS = ["আছে", "বিক্রি", "পাওয়া", "স্টক", "available"]
DETAIL_KEYWORDS = ["বিস্তারিত", "বর্ণনা", "কি", "কী", "বলুন", "জানান"]


def classify_query(query: str) -> str:
    """
    Classify query type for response strategy.
    Returns: 'price', 'availability', 'detail', or 'conversational'
    """
    q = query.lower()
    if any(kw in q for kw in PRICE_QUERY_KEYWORDS):
        return "price"
    if any(kw in q for kw in AVAILABILITY_KEYWORDS):
        return "availability"
    if any(kw in q for kw in DETAIL_KEYWORDS):
        return "detail"
    return "conversational"


def template_response(query_type: str, results: list[dict]) -> str:
    """
    Generate a template-based response from retrieved products.
    Runs in <1ms — no LLM involved.
    """
    if not results:
        return "দুঃখিত, আপনার খোঁজা পণ্যটি আমাদের কাছে নেই।"

    if query_type == "price":
        lines = []
        for p in results:
            lines.append(f"• {p['name_bn']} — মূল্য: ৳{p['price_bdt']}")
        header = f"পণ্যের মূল্য তালিকা ({len(results)}টি):"
        return f"{header}\n" + "\n".join(lines)

    elif query_type == "availability":
        names = [p["name_bn"] for p in results]
        product_list = ", ".join(names)
        return f"হ্যাঁ, আমাদের কাছে {product_list} সহ মোট {len(results)}+ ধরনের পণ্য পাওয়া যায়।"

    elif query_type == "detail":
        p = results[0]
        return (
            f"{p['name_bn']}\n"
            f"বিভাগ: {p['category_bn']}\n"
            f"মূল্য: ৳{p['price_bdt']}\n"
            f"{p['description_bn']}"
        )

    # Fallback for unclassified — still use template
    lines = []
    # Show top 5 relevant options so the user isn't forced into just 1 random product
    for p in results[:5]:
        lines.append(f"• {p['name_bn']} (৳{p['price_bdt']})")
    
    return "আপনার খোঁজা প্রাসঙ্গিক কিছু পণ্য নিচে দেওয়া হলো:\n" + "\n".join(lines)


# ─── LLM-Based Response (NOT in hot path) ───

SYSTEM_PROMPT = """আপনি একটি বাংলা ই-কমার্স সহকারী। আপনি পণ্যের তথ্য প্রদান করেন।
নিচে কিছু পণ্যের তথ্য দেওয়া হলো। এই তথ্যের ভিত্তিতে ব্যবহারকারীর প্রশ্নের উত্তর দিন।
সংক্ষিপ্ত ও সরাসরি উত্তর দিন। যদি তথ্য না পান, বলুন যে পণ্যটি পাওয়া যায়নি।"""


def llm_response(query: str, results: list[dict], history: list[dict]) -> str:
    """
    Generate an LLM-based conversational response.
    Used for Q1 and complex queries — NOT time-critical.
    """
    context_lines = []
    for i, p in enumerate(results, 1):
        context_lines.append(
            f"{i}. পণ্যের নাম: {p['name_bn']}, "
            f"বিভাগ: {p['category_bn']}, "
            f"মূল্য: ৳{p['price_bdt']}, "
            f"বিবরণ: {p['description_bn']}"
        )
    context = "\n".join(context_lines)

    user_prompt = f"পণ্যের তথ্য:\n{context}\n\nব্যবহারকারীর প্রশ্ন: {query}"

    return groq_llm.generate_with_history(
        system_prompt=SYSTEM_PROMPT,
        history=history,
        user_query=user_prompt,
        max_tokens=256,
    )
