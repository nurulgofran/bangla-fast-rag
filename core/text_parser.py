"""
Text parser for unstructured Bangla product data.
Reads freeform text paragraphs and extracts structured product fields
using regex-based metadata extraction (industry-standard for unstructured ingestion).
"""
import re
from pathlib import Path


def parse_products(filepath: str | Path) -> list[dict]:
    """
    Parse a plain-text product file into structured dicts.

    Expected format: one product per paragraph, separated by blank lines.
    Each paragraph follows the pattern:
        <name> — <category>। মূল্য: ৳<price> টাকা। <description>

    Returns:
        list[dict] with keys: name_bn, category_bn, price_bdt, description_bn
    """
    filepath = Path(filepath)
    raw = filepath.read_text(encoding="utf-8")

    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]

    products = []
    for idx, para in enumerate(paragraphs, start=1):
        product = _parse_paragraph(para, idx)
        if product:
            products.append(product)

    return products


def _parse_paragraph(text: str, product_id: int) -> dict | None:
    """Extract structured fields from a single product paragraph."""

    # Extract name: everything before the em dash
    name_match = re.match(r'^(.+?)\s*—\s*', text)
    if not name_match:
        return None
    name = name_match.group(1).strip()

    # Extract category: text between em dash and first period
    cat_match = re.search(r'—\s*(.+?)।', text)
    category = cat_match.group(1).strip() if cat_match else ""

    # Extract price: মূল্য: ৳<digits>
    price_match = re.search(r'মূল্য:\s*৳(\d+)', text)
    price = int(price_match.group(1)) if price_match else 0

    # Extract description: everything after the price sentence
    desc_match = re.search(r'মূল্য:\s*৳\d+\s*টাকা।\s*(.+)', text, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else text

    return {
        "id": product_id,
        "name_bn": name,
        "category_bn": category,
        "price_bdt": price,
        "description_bn": description,
    }
