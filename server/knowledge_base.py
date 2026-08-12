"""Knowledge Base management and RAG retrieval for the Conversational AI Playground.

Handles:
- CRUD operations on knowledge bases and their documents
- Document text extraction (txt, md, pdf)
- Text chunking with overlap
- Embedding generation via Google text-embedding-004
- Cosine-similarity retrieval for RAG context injection
"""

import json
import math
import os
import uuid
from pathlib import Path

from loguru import logger

from db import KB_FILES_DIR, get_connection

# ── Constants ──────────────────────────────────────────────────────────
CHUNK_SIZE = 500          # target tokens per chunk (~4 chars/token heuristic)
CHUNK_OVERLAP = 50        # overlap tokens between consecutive chunks
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMS = 768      # text-embedding-004 output dimension


# ══════════════════════════════════════════════════════════════════════
# Knowledge Base CRUD
# ══════════════════════════════════════════════════════════════════════

def create_kb(name: str, description: str = "") -> dict:
    """Create a new knowledge base."""
    kb_id = uuid.uuid4().hex[:12]
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO knowledge_bases (id, name, description) VALUES (?, ?, ?)",
            (kb_id, name, description),
        )
        conn.commit()
        kb_dir = KB_FILES_DIR / kb_id
        kb_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created knowledge base '{name}' (id={kb_id})")
        return {"id": kb_id, "name": name, "description": description}
    finally:
        conn.close()


def list_kbs() -> list[dict]:
    """List all knowledge bases with document counts."""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT kb.id, kb.name, kb.description, kb.created_at, kb.updated_at,
                   COUNT(d.id) AS doc_count,
                   COALESCE(SUM(d.chunk_count), 0) AS total_chunks
            FROM knowledge_bases kb
            LEFT JOIN documents d ON d.kb_id = kb.id AND d.status = 'ready'
            GROUP BY kb.id
            ORDER BY kb.created_at DESC
        """).fetchall()
        kbs = [dict(r) for r in rows]
        if not kbs:
            # Auto-seed sample KBs if database is empty
            seed_sample_kbs()
            rows = conn.execute("""
                SELECT kb.id, kb.name, kb.description, kb.created_at, kb.updated_at,
                       COUNT(d.id) AS doc_count,
                       COALESCE(SUM(d.chunk_count), 0) AS total_chunks
                FROM knowledge_bases kb
                LEFT JOIN documents d ON d.kb_id = kb.id AND d.status = 'ready'
                GROUP BY kb.id
                ORDER BY kb.created_at DESC
            """).fetchall()
            kbs = [dict(r) for r in rows]
        return kbs
    finally:
        conn.close()


def seed_sample_kbs():
    """Seed sample knowledge bases for all playground scenarios if they don't already exist."""
    conn = get_connection()
    try:
        count = conn.execute("SELECT count(*) FROM knowledge_bases").fetchone()[0]
        if count > 0:
            return
    finally:
        conn.close()

    samples = [
        {
            "name": "Spice Garden Restaurant",
            "desc": "Menu, opening hours, prices, dietary options, address, and policies for Spice Garden Bistro.",
            "filename": "spice_garden_menu.md",
            "content": """# Spice Garden Restaurant & Bistro — Information & Menu
## 📍 General Information
- **Restaurant Name**: Spice Garden Restaurant & Bistro
- **Cuisine**: Modern Indian & Fusion
- **Address**: 42 Culinary Avenue, Gourmet District, Bangalore 560001
- **Phone**: +91 80 4567 8900
- **Email**: reservations@spicegardenbistro.com
- **Website**: www.spicegardenbistro.com

## ⏰ Opening Hours
- **Monday – Thursday**: 12:00 PM – 3:30 PM (Lunch), 7:00 PM – 11:00 PM (Dinner)
- **Friday – Sunday**: 12:00 PM – 11:30 PM (Continuous Dining)
- **Happy Hours**: Mon–Fri 4:00 PM – 7:00 PM (50% off select cocktails & craft beers)

## 🥗 Starters & Appetizers
1. **Truffle Malai Paneer Tikka** – ₹420
   - Creamy cottage cheese marinated in cardamom, white pepper, and black truffle oil, charcoal-grilled in tandoor. *(Gluten-Free, Vegetarian)*
2. **Kolkata Spiced Prawn Chettinad Tacos** – ₹540
   - Crispy mini parathas filled with pan-seared prawns, roasted coconut spices, and curry leaf aioli.
3. **Gunpowder Jackfruit Sliders** – ₹380
   - Pulled tender jackfruit tossed in Andhra gunpowder ghee, served in toasted brioche buns. *(Vegetarian)*

## 🍲 Main Course
1. **Old Delhi Butter Chicken (Classic)** – ₹580
   - Overnight marinated chicken tikka simmered in a velvet tomato, cashew, and fenugreek gravy. Served with garlic naan.
2. **Nizamabad Mutton Dum Biryani** – ₹690
   - Fragrant long-grain basmati rice cooked on slow dum with tender goat meat, saffron, and aromatic spices. Served with burani raita.
3. **Kerala Raw Mango & Fish Curry** – ₹620
   - Kingfish simmered in a tangy coconut milk broth with raw mango slices and tempered mustard seeds. *(Gluten-Free)*

## 🍹 Signature Cocktails & Beverages
1. **Deccan Sunset** – ₹450
   - Single malt whiskey infused with star anise, jaggery syrup, and orange bitters.
2. **Masala Chai Martini** – ₹400
   - Vodka, freshly brewed spiced Assam tea, Kahlua, and a dash of cardamom.
3. **Kachha Aam Cooler (Mocktail)** – ₹220
   - Raw green mango pulp, roasted cumin, mint leaves, and sparkling soda.

## 📋 Policies & Information
- **Reservations**: Recommended on weekends. Tables held for up to 15 minutes.
- **Dietary Options**: Vegan, Jain, and Gluten-Free dishes clearly marked on menu.
- **Valet Parking**: Complimentary valet parking available at restaurant entrance.
- **Private Dining**: Hall available for private events up to 25 guests."""
        },
        {
            "name": "Grand Meridian Hotel & Resort",
            "desc": "Room rates, check-in policies, amenities, dining, spa, and shuttle service details.",
            "filename": "grand_meridian_guide.md",
            "content": """# Grand Meridian Hotel & Resort — Guest Directory

## 🏨 Hotel Overview
- **Hotel Name**: The Grand Meridian Hotel & Resort
- **Address**: 100 Ocean Drive, Marine District, Mumbai 400021
- **Phone**: +91 22 6789 0000
- **Email**: reservations@grandmeridianhotel.com
- **Website**: www.grandmeridianhotel.com

## 🛌 Room Types & Rates (Breakfast Included)
1. **Deluxe Room** — ₹6,500 ($79) / night
   - 350 sq.ft, King or Twin beds, City view, Smart TV, Work desk, Marble bathroom.
2. **Executive Suite** — ₹11,000 ($132) / night
   - 650 sq.ft, Sea view, Living room, King bed, Nespresso machine, Executive Lounge access.
3. **Family Room** — ₹9,000 ($108) / night
   - 500 sq.ft, Two Queen beds, Connecting room option, Kid-friendly amenities.

## ⏰ Check-In & Check-Out
- **Check-In Time**: 2:00 PM
- **Check-Out Time**: 11:00 AM
- **Early Check-In / Late Check-Out**: Subject to availability (Complimentary up to 2 hours for Gold members).

## 🏊 Amenities & Services
- **Wi-Fi**: High-speed complimentary Wi-Fi across hotel premises.
- **Swimming Pool & Fitness Center**: Open daily 6:00 AM – 10:00 PM (Complimentary for guests).
- **Lotus Spa**: Open daily 9:00 AM – 9:00 PM (Aromatherapy, Ayurvedic massages, facial treatments).
- **Airport Shuttle**: Available 24/7 upon request (₹1,200 / $15 per vehicle per trip).
- **Dining**: Meridian All-Day Dining Restaurant (6:30 AM – 11:00 PM) & Sky Lounge Bar (5:00 PM – 1:00 AM)."""
        },
        {
            "name": "Sunrise Multi-Speciality Hospital",
            "desc": "Doctor consultations, department specialities, insurance coverage, emergency, and visiting hours.",
            "filename": "sunrise_hospital_guide.md",
            "content": """# Sunrise Multi-Speciality Hospital — Patient Guide

## 🏥 Hospital Overview
- **Hospital Name**: Sunrise Multi-Speciality Hospital
- **Address**: 88 Health Park Way, Indiranagar, Bangalore 560038
- **Helpline / Appointments**: +91 80 9876 5432
- **Emergency Helpline**: 1066 (24/7 Ambulance & Trauma Care)
- **Email**: care@sunrisemultispecialityhospital.com
- **OPD Timings**: Monday – Saturday: 8:00 AM – 8:00 PM

## 🩺 Departments & Specialists
1. **Cardiology & Cardiothoracic Surgery**
   - **Dr. Rajesh Sharma, MD, DM**: Mon, Wed, Fri (9:00 AM – 1:00 PM). Consultation Fee: ₹1,000.
2. **Pediatrics & Child Care**
   - **Dr. Meera Patel, DCH, MD**: Tue, Thu, Sat (10:00 AM – 4:00 PM). Consultation Fee: ₹800.
3. **Orthopedics & Joint Replacement**
   - **Dr. Vikram Rao, MS (Ortho)**: Mon–Sat (2:00 PM – 6:00 PM). Consultation Fee: ₹900.
4. **Neurology & Stroke Clinic**
   - **Dr. Ananya Roy, DM (Neuro)**: Tue, Thu (11:00 AM – 3:00 PM). Consultation Fee: ₹1,200.

## 💳 Insurance & Cashless Desk
- **Cashless Empanelled TPA / Insurance Providers**: Star Health, HDFC ERGO, Max Bupa, ICICI Lombard, Care Health, Raksha TPA, Medi Assist.
- **Cashless Helpdesk Hours**: 24/7 at Ground Floor Desk 4.

## 🕒 Visiting Hours & Guidelines
- **General Wards**: 4:00 PM – 7:00 PM daily (Max 2 visitors per patient).
- **ICU / CCU**: 11:00 AM – 12:00 PM & 5:00 PM – 6:00 PM (Strictly 1 attendant permitted)."""
        },
        {
            "name": "Apex Global Bank & Wealth",
            "desc": "Savings accounts, fixed deposits, interest rates, loans, credit cards, and customer care details.",
            "filename": "apex_bank_services.md",
            "content": """# Apex Global Bank & Wealth — Banking & Product Guide

## 🏦 Bank Overview
- **Bank Name**: Apex Global Bank
- **Toll-Free Customer Support**: 1800-123-4567 (24/7)
- **Email**: support@apexglobalbank.com
- **Net Banking**: www.apexglobalbank.com
- **Head Office**: Apex Financial Towers, MG Road, Bangalore 560001

## 💰 Accounts & Deposits
1. **Apex Signature Savings Account**
   - Interest Rate: 4.0% p.a. (up to ₹1 Lakh), 6.5% p.a. (above ₹1 Lakh). Minimum balance: ₹10,000.
2. **Fixed Deposits (FD)**
   - 1 Year: 7.20% p.a. (Senior Citizens: 7.70% p.a.)
   - 3 Years: 7.80% p.a. (Senior Citizens: 8.30% p.a.)
   - Tax Saver FD (5 Years): 7.50% p.a. with Section 80C tax benefits.

## 💳 Credit Cards & Loans
1. **Apex Platinum Rewards Credit Card**
   - Annual Fee: ₹499 (Waived on ₹50,000 annual spend). 4x Reward Points on dining & travel.
2. **Home Loans**
   - Floating Interest Rate: Starting at 8.35% p.a. Tenure up to 30 years. Zero processing fee for women applicants.
3. **Personal Loans**
   - Instant approval up to ₹15 Lakhs. Interest rate: 10.5% – 14.0% p.a.

## 🛡️ Security & Support
- **Report Lost Card / Block Card**: Call 1800-123-4567 or SMS `BLOCK <Last 4 digits>` to 56767.
- **ATM Daily Cash Withdrawal Limit**: ₹50,000 (Platinum Card ₹1,000,000)."""
        },
        {
            "name": "NovaCart Online Superstore",
            "desc": "Shipping timelines, delivery rates, 30-day return policy, order tracking, and warranty claim steps.",
            "filename": "novacart_customer_help.md",
            "content": """# NovaCart Online Superstore — Delivery & Return Policy

## 🛒 Customer Care Overview
- **Store Name**: NovaCart Online Superstore
- **Toll-Free Support**: 1800-888-9999 (24/7 Customer Hotline)
- **Email Support**: help@novacart.com
- **Live Chat**: Available 24/7 on website and mobile app

## 🚚 Shipping & Delivery Rates
- **Standard Delivery**: 3–5 business days. Free shipping on all orders over ₹499 (Flat ₹50 fee for orders under ₹499).
- **Express Same-Day Delivery**: Delivered within 12–24 hours in select metropolitan cities for flat ₹149.
- **Order Tracking**: Track your parcel live at www.novacart.com/track using your 10-digit Order ID.

## 🔄 Return & Refund Policy
- **30-Day Easy Returns**: Unused items in original packaging with tags intact can be returned within 30 days of delivery.
- **Pick-up**: Free reverse pick-up scheduled at your doorstep within 48 hours of return request.
- **Refund Processing**: Refunds processed back to original payment mode within 5–7 business days after item inspection.

## 🔧 Warranty & Damage Claims
- **Damaged or Missing Item**: Report via app within 48 hours of delivery with photo proof for instant replacement.
- **Brand Warranty**: Electronics carry 1–2 year manufacturer warranty. Digital invoice available in account dashboard serves as proof of purchase."""
        }
    ]

    for s in samples:
        try:
            kb = create_kb(s["name"], s["desc"])
            kb_id = kb["id"]
            doc_id = uuid.uuid4().hex[:12]
            filename = s["filename"]
            content_text = s["content"]
            content_bytes = content_text.encode("utf-8")

            # Save physical file
            kb_dir = KB_FILES_DIR / kb_id
            kb_dir.mkdir(parents=True, exist_ok=True)
            file_path = kb_dir / f"{doc_id}.md"
            file_path.write_bytes(content_bytes)

            c = get_connection()
            try:
                c.execute(
                    """INSERT INTO documents (id, kb_id, filename, file_size, content_type, chunk_count, status)
                       VALUES (?, ?, ?, ?, ?, ?, 'ready')""",
                    (doc_id, kb_id, filename, len(content_bytes), "text/markdown", 0),
                )
                c.commit()

                # Chunk & TF-IDF Embed synchronously
                chunks = _chunk_text(content_text)
                embeddings = _tfidf_embeddings(chunks)

                for i, chunk_text in enumerate(chunks):
                    chunk_id = uuid.uuid4().hex[:12]
                    emb_json = json.dumps(embeddings[i])
                    token_count = math.ceil(len(chunk_text) / 4)
                    c.execute(
                        """INSERT INTO chunks (id, doc_id, kb_id, chunk_index, text, embedding, token_count)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (chunk_id, doc_id, kb_id, i, chunk_text, emb_json, token_count),
                    )

                c.execute(
                    "UPDATE documents SET chunk_count = ?, status = 'ready' WHERE id = ?",
                    (len(chunks), doc_id),
                )
                c.commit()
                logger.info(f"Seeded sample KB: '{s['name']}' (id={kb_id})")
            finally:
                c.close()
        except Exception as e:
            logger.warning(f"Failed to seed sample KB '{s['name']}': {e}")


def get_kb(kb_id: str) -> dict | None:
    """Get a single knowledge base."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_kb(kb_id: str, name: str, description: str = "") -> dict | None:
    """Update a knowledge base's name and description."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE knowledge_bases SET name = ?, description = ? WHERE id = ?",
            (name, description, kb_id),
        )
        conn.commit()
        return get_kb(kb_id)
    finally:
        conn.close()


def delete_kb(kb_id: str) -> bool:
    """Delete a knowledge base and all its documents/chunks."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
        conn.commit()
        # Remove files
        import shutil
        kb_dir = KB_FILES_DIR / kb_id
        if kb_dir.exists():
            shutil.rmtree(kb_dir)
        logger.info(f"Deleted knowledge base id={kb_id}")
        return True
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════
# Document Management
# ══════════════════════════════════════════════════════════════════════

def list_documents(kb_id: str) -> list[dict]:
    """List all documents in a knowledge base."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM documents WHERE kb_id = ? ORDER BY created_at DESC",
            (kb_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_document_details(kb_id: str, doc_id: str) -> dict | None:
    """Get document details including its chunks and text."""
    conn = get_connection()
    try:
        doc = conn.execute(
            "SELECT * FROM documents WHERE id = ? AND kb_id = ?",
            (doc_id, kb_id),
        ).fetchone()
        if not doc:
            return None

        chunks = conn.execute(
            "SELECT chunk_index, text, token_count FROM chunks WHERE doc_id = ? ORDER BY chunk_index ASC",
            (doc_id,),
        ).fetchall()

        doc_dict = dict(doc)
        doc_dict["chunks"] = [dict(c) for c in chunks]
        return doc_dict
    finally:
        conn.close()


async def update_document_text(kb_id: str, doc_id: str, new_text: str) -> dict | None:
    """Update a document's text, re-chunk, and re-embed."""
    conn = get_connection()
    try:
        doc = conn.execute("SELECT * FROM documents WHERE id = ? AND kb_id = ?", (doc_id, kb_id)).fetchone()
        if not doc:
            return None

        # Delete old chunks
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.commit()

        # Update file on disk if plain text/markdown
        file_path = KB_FILES_DIR / kb_id / f"{doc_id}{Path(doc['filename']).suffix or '.txt'}"
        if file_path.exists():
            file_path.write_text(new_text, encoding="utf-8")

        # Chunk text
        chunks = _chunk_text(new_text)

        # Generate embeddings
        embeddings = await _embed_texts(chunks)

        # Insert new chunks
        for i, chunk_text in enumerate(chunks):
            chunk_id = uuid.uuid4().hex[:12]
            emb_json = json.dumps(embeddings[i])
            token_count = math.ceil(len(chunk_text) / 4)
            conn.execute(
                """INSERT INTO chunks (id, doc_id, kb_id, chunk_index, text, embedding, token_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, doc_id, kb_id, i, chunk_text, emb_json, token_count),
            )

        conn.execute(
            "UPDATE documents SET file_size = ?, chunk_count = ?, status = 'ready', error_message = NULL WHERE id = ?",
            (len(new_text.encode("utf-8")), len(chunks), doc_id),
        )
        conn.commit()
        logger.info(f"Updated document id={doc_id}: {len(chunks)} chunks re-embedded")
        return get_document_details(kb_id, doc_id)
    except Exception as e:
        conn.execute("UPDATE documents SET status = 'error', error_message = ? WHERE id = ?", (str(e), doc_id))
        conn.commit()
        raise e
    finally:
        conn.close()


def duplicate_kb(kb_id: str) -> dict | None:
    """Duplicate/clone an existing knowledge base and all its documents and chunks."""
    source_kb = get_kb(kb_id)
    if not source_kb:
        return None

    new_name = f"{source_kb['name']} (Copy)"
    new_kb = create_kb(new_name, source_kb.get("description", ""))
    new_kb_id = new_kb["id"]

    conn = get_connection()
    try:
        source_docs = conn.execute("SELECT * FROM documents WHERE kb_id = ?", (kb_id,)).fetchall()
        for doc in source_docs:
            new_doc_id = uuid.uuid4().hex[:12]
            conn.execute(
                """INSERT INTO documents (id, kb_id, filename, file_size, content_type, chunk_count, status, error_message)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (new_doc_id, new_kb_id, doc["filename"], doc["file_size"], doc["content_type"], doc["chunk_count"], doc["status"], doc["error_message"]),
            )
            # Duplicate chunks
            chunks = conn.execute("SELECT * FROM chunks WHERE doc_id = ?", (doc["id"],)).fetchall()
            for c in chunks:
                new_chunk_id = uuid.uuid4().hex[:12]
                conn.execute(
                    """INSERT INTO chunks (id, doc_id, kb_id, chunk_index, text, embedding, token_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (new_chunk_id, new_doc_id, new_kb_id, c["chunk_index"], c["text"], c["embedding"], c["token_count"]),
                )

            # Duplicate physical file on disk if exists
            import shutil
            src_dir = KB_FILES_DIR / kb_id
            dst_dir = KB_FILES_DIR / new_kb_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src_file in src_dir.glob(f"{doc['id']}.*"):
                dst_file = dst_dir / f"{new_doc_id}{src_file.suffix}"
                shutil.copy2(src_file, dst_file)

        conn.commit()
        logger.info(f"Duplicated KB id={kb_id} -> new_id={new_kb_id}")
        return get_kb(new_kb_id)
    finally:
        conn.close()


def rename_document(kb_id: str, doc_id: str, new_filename: str) -> dict | None:
    """Rename a document in a knowledge base."""
    new_filename = new_filename.strip()
    if not new_filename:
        return None
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE documents SET filename = ? WHERE id = ? AND kb_id = ?",
            (new_filename, doc_id, kb_id),
        )
        conn.commit()
        logger.info(f"Renamed document id={doc_id} in kb={kb_id} to '{new_filename}'")
        return get_document_details(kb_id, doc_id)
    finally:
        conn.close()


def delete_document(kb_id: str, doc_id: str) -> bool:
    """Delete a document and its chunks from a knowledge base."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM documents WHERE id = ? AND kb_id = ?", (doc_id, kb_id))
        conn.commit()
        # Remove file
        kb_dir = KB_FILES_DIR / kb_id
        for f in kb_dir.glob(f"{doc_id}.*"):
            f.unlink(missing_ok=True)
        logger.info(f"Deleted document id={doc_id} from kb={kb_id}")
        return True
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════
# Text Extraction
# ══════════════════════════════════════════════════════════════════════

def _extract_text(file_path: Path, content_type: str) -> str:
    """Extract plain text from a file based on its type."""
    suffix = file_path.suffix.lower()

    if suffix in (".txt", ".md", ".markdown"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            import pymupdf
            text_parts = []
            with pymupdf.open(str(file_path)) as doc:
                for page in doc:
                    text_parts.append(page.get_text())
            return "\n\n".join(text_parts)
        except ImportError:
            logger.error("pymupdf not installed — cannot extract PDF text")
            raise ValueError("PDF support requires pymupdf. Install with: uv add pymupdf")
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")

    raise ValueError(f"Unsupported file type: {suffix}")


# ══════════════════════════════════════════════════════════════════════
# Text Chunking
# ══════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of roughly chunk_size tokens with overlap.

    Uses a simple heuristic: ~4 characters per token. Breaks at sentence
    boundaries (period, newline) when possible.
    """
    if not text.strip():
        return []

    char_chunk = chunk_size * 4
    char_overlap = overlap * 4

    # Split into sentences (rough: split on .\n or \n\n or . followed by space+uppercase)
    import re
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > char_chunk and current:
            chunks.append(current.strip())
            # Keep overlap from the end of the current chunk
            if char_overlap > 0 and len(current) > char_overlap:
                current = current[-char_overlap:] + " " + sentence
            else:
                current = sentence
        else:
            current = current + " " + sentence if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ══════════════════════════════════════════════════════════════════════
# Embedding Generation
# ══════════════════════════════════════════════════════════════════════

def _tfidf_embeddings(texts: list[str]) -> list[list[float]]:
    """Pure-Python TF-IDF embedding generator fallback.

    Creates normalized term-frequency vectors over the vocabulary of the
    texts so cosine similarity retrieval works robustly offline or when
    API key embedding permissions are restricted.
    """
    import re
    from collections import Counter

    def tokenize(t):
        return re.findall(r'\w+', t.lower())

    docs_tokens = [tokenize(t) for t in texts]
    # Build vocabulary
    vocab = sorted(set(w for doc in docs_tokens for w in doc if len(w) > 1))
    if not vocab:
        return [[0.0] * 10 for _ in texts]

    # Calculate IDF
    N = len(texts)
    doc_freq = Counter()
    for doc in docs_tokens:
        for word in set(doc):
            doc_freq[word] += 1

    idf = {word: math.log((N + 1) / (freq + 1)) + 1.0 for word, freq in doc_freq.items()}

    # Calculate normalized TF-IDF vectors
    vectors = []
    for doc in docs_tokens:
        tf = Counter(doc)
        doc_len = max(len(doc), 1)
        vec = [ (tf[w] / doc_len) * idf.get(w, 0.0) for w in vocab ]
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        vectors.append(vec)

    return vectors


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using Google Generative AI,
    falling back to TF-IDF vectorization if the API key lacks embedding permissions.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            import google.genai as genai
            client = genai.Client(api_key=api_key)
            embeddings = []
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Try gemini-embedding-001 or text-embedding-004
                result = await client.aio.models.embed_content(
                    model="text-embedding-004",
                    contents=batch,
                )
                for emb in result.embeddings:
                    embeddings.append(emb.values)
            if len(embeddings) == len(texts):
                return embeddings
        except Exception as e:
            logger.warning(f"Google AI embedding API unavailable ({e}), using TF-IDF fallback")

    # Fallback to pure-python TF-IDF vectors
    return _tfidf_embeddings(texts)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ══════════════════════════════════════════════════════════════════════
# Document Upload Pipeline
# ══════════════════════════════════════════════════════════════════════

async def upload_document(kb_id: str, filename: str, content: bytes) -> dict:
    """Process and store a document in a knowledge base.

    Pipeline: save file → extract text → chunk → embed → store in DB.
    """
    doc_id = uuid.uuid4().hex[:12]
    suffix = Path(filename).suffix.lower()
    content_type = {
        ".txt": "text/plain", ".md": "text/markdown",
        ".markdown": "text/markdown", ".pdf": "application/pdf",
    }.get(suffix, "text/plain")

    # Save the file
    kb_dir = KB_FILES_DIR / kb_id
    kb_dir.mkdir(parents=True, exist_ok=True)
    file_path = kb_dir / f"{doc_id}{suffix}"
    file_path.write_bytes(content)

    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO documents (id, kb_id, filename, file_size, content_type, status)
               VALUES (?, ?, ?, ?, ?, 'processing')""",
            (doc_id, kb_id, filename, len(content), content_type),
        )
        conn.commit()
    finally:
        conn.close()

    try:
        # Extract text
        text = _extract_text(file_path, content_type)
        if not text.strip():
            raise ValueError("No text content could be extracted from the file")

        # Chunk
        chunks = _chunk_text(text)
        if not chunks:
            raise ValueError("Document produced no text chunks")

        logger.info(f"Document '{filename}': extracted {len(text)} chars, {len(chunks)} chunks")

        # Embed all chunks
        embeddings = await _embed_texts(chunks)

        # Store chunks with embeddings
        conn = get_connection()
        try:
            for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = uuid.uuid4().hex[:12]
                conn.execute(
                    """INSERT INTO chunks (id, doc_id, kb_id, chunk_index, text, embedding, token_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (chunk_id, doc_id, kb_id, idx, chunk_text,
                     json.dumps(embedding), len(chunk_text) // 4),
                )

            conn.execute(
                "UPDATE documents SET chunk_count = ?, status = 'ready' WHERE id = ?",
                (len(chunks), doc_id),
            )
            conn.execute(
                "UPDATE knowledge_bases SET updated_at = datetime('now') WHERE id = ?",
                (kb_id,),
            )
            conn.commit()
            logger.info(f"Document '{filename}' ready: {len(chunks)} chunks embedded")
        finally:
            conn.close()

        return {
            "id": doc_id, "kb_id": kb_id, "filename": filename,
            "chunk_count": len(chunks), "status": "ready",
        }

    except Exception as e:
        logger.error(f"Document processing failed for '{filename}': {e}")
        conn = get_connection()
        try:
            conn.execute(
                "UPDATE documents SET status = 'error', error_message = ? WHERE id = ?",
                (str(e)[:500], doc_id),
            )
            conn.commit()
        finally:
            conn.close()
        return {
            "id": doc_id, "kb_id": kb_id, "filename": filename,
            "chunk_count": 0, "status": "error", "error": str(e),
        }


# ══════════════════════════════════════════════════════════════════════
# RAG Retrieval
# ══════════════════════════════════════════════════════════════════════

async def retrieve(kb_ids: list[str], query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the most relevant chunks from the specified knowledge bases.

    Embeds the query, then searches all chunks in the given KBs by cosine
    similarity or term overlap. Returns the top-k results with text and score.
    """
    if not kb_ids or not query.strip():
        return []

    # Fetch all chunk embeddings from the target KBs
    conn = get_connection()
    try:
        placeholders = ",".join("?" for _ in kb_ids)
        rows = conn.execute(
            f"""SELECT c.id, c.text, c.embedding, c.kb_id, d.filename
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE c.kb_id IN ({placeholders})
                  AND d.status = 'ready'""",
            kb_ids,
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    chunk_texts = [r["text"] for r in rows]

    # Try embedding the query alongside chunk texts for accurate TF-IDF/semantic scoring
    embeddings = await _embed_texts([query] + chunk_texts)
    query_emb = embeddings[0]
    chunk_embs = embeddings[1:]

    # Score each chunk using cosine similarity + query term coverage boost
    import re
    query_words = set(re.findall(r'\w+', query.lower())) - {'a', 'an', 'the', 'is', 'are', 'what', 'whats', 'what\'s', 'your', 'my', 'of', 'in', 'on', 'for', 'to', 'and', 'do', 'you', 'have', 'tell', 'me', 'about', 'can', 'i', 'get'}

    scored = []
    for i, row in enumerate(rows):
        emb = chunk_embs[i]
        sim_score = _cosine_similarity(query_emb, emb)

        # Keyword coverage score
        chunk_words = set(re.findall(r'\w+', row["text"].lower()))
        if query_words:
            overlap = len(query_words.intersection(chunk_words))
            coverage = overlap / len(query_words)
        else:
            coverage = 0.0

        # Combined score (similarity + keyword coverage boost)
        final_score = (sim_score * 0.5) + (coverage * 0.5) if query_words else sim_score

        scored.append({
            "chunk_id": row["id"],
            "text": row["text"],
            "score": round(final_score, 4),
            "kb_id": row["kb_id"],
            "filename": row["filename"],
        })

    # Sort by score descending, return top-k
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def build_rag_context(passages: list[dict]) -> str:
    """Format retrieved passages into a context block for the system prompt."""
    if not passages:
        return ""

    lines = [
        "=== KNOWLEDGE BASE GROUNDING INSTRUCTION ===",
        "CRITICAL: A Knowledge Base has been explicitly selected for this conversation.",
        "You MUST answer the user's questions using the facts provided in the Knowledge Base passages below.",
        "When answering questions about contact details, email, phone number, address, operating hours, menu, items, prices, or policies, "
        "rely STRICTLY on the Knowledge Base passages provided here.",
        "If a default scenario persona (such as hotel, hospital, or bank) conflicts with the Knowledge Base, "
        "the Knowledge Base information ALWAYS takes priority.",
        "",
    ]

    for i, p in enumerate(passages, 1):
        source = p.get("filename", "unknown")
        lines.append(f"--- Knowledge Base Passage {i} (Source: {source}, Relevance: {p['score']:.2f}) ---")
        lines.append(p["text"])
        lines.append("")

    lines.append("=== END OF KNOWLEDGE BASE GROUNDING INSTRUCTION ===")
    return "\n".join(lines)
