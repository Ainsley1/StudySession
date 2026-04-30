"""
AI Archive Bot v4 — Handles scanned PDFs + better section detection
Key fixes:
1. Scanned PDF fallback: estimates question counts from page counts
2. Broader question number regex catches more formats  
3. Smarter section detection: skips cover/instruction pages
4. Page-based extraction always works even when text extraction fails
"""

import os, re, json, logging, requests, tempfile
from datetime import datetime
from pypdf import PdfReader, PdfWriter
import pdfplumber
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "")
ALLOWED_USER_ID = int(os.environ.get("ALLOWED_USER_ID", "0"))

KNOWN_USERS = {}  # "telegramusername": "Display Name"


# ─── SECTION DETECTION ───────────────────────────────────────────────
MCQ_MARKERS = [
    "section a", "part a", "multiple choice", "booklet a",
    "questions 1 to", "choose the correct", "shade the correct",
    "circle the correct", "for each question",
    "each question carries 1 mark", "each carries 1 mark",
]
OE_MARKERS = [
    "section b", "part b", "open-ended", "open ended", "booklet b",
    "structured question", "short answer",
    "write your answer in the space",
    "each question carries 2", "each question carries 3",
    "each question carries 4", "carries 2 marks", "carries 3 marks",
]
ANS_MARKERS = [
    "answer key", "marking scheme", "suggested answer",
    "marking guide", "answers to", "model answer",
]
SKIP_MARKERS = [  # pages to ignore (cover, instructions, blank)
    "do not open this booklet",
    "name:", "class:", "register",
    "instructions to candidates",
    "write your name",
    "this paper consists of",
    "end of paper",
    "— end —", "* end *",
    "blank page", "this page is intentionally",
]

# Typical SG exam questions per page (for scanned PDF estimation)
MCQ_PER_PAGE = {(1,3): 8, (4,6): 6, (7,20): 5}   # pages: q_per_page
OE_PER_PAGE  = {(1,4): 4, (5,10): 3, (11,30): 2}


def estimate_q_count(n_pages: int, section: str) -> int:
    """Estimate question count from page count when PDF is scanned."""
    if n_pages == 0:
        return 0
    if section == "MCQ":
        for (lo, hi), ppg in MCQ_PER_PAGE.items():
            if lo <= n_pages <= hi:
                return n_pages * ppg
        return n_pages * 5
    else:
        for (lo, hi), ppg in OE_PER_PAGE.items():
            if lo <= n_pages <= hi:
                return n_pages * ppg
        return n_pages * 2


def detect_sections(pdf_path: str) -> dict:
    """
    Two-pass detection:
    Pass 1: Find section boundaries from headers
    Pass 2: Find question numbers within each section
    Falls back to page estimates if PDF is scanned (no extractable text)
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    mcq_pages, oe_pages, ans_pages = [], [], []
    mcq_q_map, oe_q_map = {}, {}
    current_section = None
    text_found = False  # track if any text was extractable

    # Broader question number patterns
    Q_PATTERNS = [
        r'(?:^|\n)\s{0,8}(\d{1,2})[\.\)\]]\s',       # 1. 1) 1]
        r'(?:^|\n)\s{0,8}(\d{1,2})\t',                # 1\t
        r'(?:^|\n)\s{0,8}(\d{1,2})\s{2,}',            # 1   (2+ spaces)
        r'(?:^|\n)\s{0,8}[Qq](?:uestion)?\s*\.?\s*(\d{1,2})', # Q1 Question 1
        r'(?:^|\n)\s{0,8}\[(\d{1,2})\]',              # [1]
        r'(?:^|\n)\s{0,8}(\d{1,2})\s*\n\s*[A-Z\(]',  # number then capital
    ]

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text_lower = text.lower().strip()

            if len(text_lower) > 20:
                text_found = True

            # Skip cover/instruction pages
            if any(kw in text_lower for kw in SKIP_MARKERS) and page_idx < 3:
                continue

            # Answer pages
            if any(kw in text_lower for kw in ANS_MARKERS):
                ans_pages.append(page_idx)
                current_section = "ANS"
                continue

            if current_section == "ANS":
                ans_pages.append(page_idx)
                continue

            # Section transitions
            if any(kw in text_lower for kw in MCQ_MARKERS):
                current_section = "MCQ"
            elif any(kw in text_lower for kw in OE_MARKERS):
                current_section = "OE"

            # Assign page to section
            if current_section == "MCQ":
                mcq_pages.append(page_idx)
            elif current_section == "OE":
                oe_pages.append(page_idx)
            elif current_section is None and page_idx > 0:
                # Unknown section after cover — assume MCQ for primary papers
                mcq_pages.append(page_idx)

            # Extract question numbers
            found = set()
            for pat in Q_PATTERNS:
                for m in re.findall(pat, text, re.MULTILINE):
                    if m.isdigit() and 1 <= int(m) <= 60:
                        found.add(int(m))

            if current_section == "MCQ":
                for q in found:
                    if q not in mcq_q_map:
                        mcq_q_map[q] = page_idx
            elif current_section == "OE":
                for q in found:
                    if q not in oe_q_map:
                        oe_q_map[q] = page_idx

    # Deduplicate and sort
    mcq_pages = sorted(set(mcq_pages))
    oe_pages  = sorted(set(oe_pages))
    ans_pages = sorted(set(ans_pages))

    # Determine question counts
    if mcq_q_map:
        mcq_total = max(mcq_q_map.keys())
    else:
        # Scanned PDF — estimate from page count
        mcq_total = estimate_q_count(len(mcq_pages), "MCQ")
        logger.info(f"Scanned PDF: estimating {mcq_total} MCQ from {len(mcq_pages)} pages")

    if oe_q_map:
        oe_total = max(oe_q_map.keys())
    else:
        oe_total = estimate_q_count(len(oe_pages), "OE")
        logger.info(f"Scanned PDF: estimating {oe_total} OE from {len(oe_pages)} pages")

    # If section detection completely failed, use Gemini
    if not mcq_pages and not oe_pages and not ans_pages:
        return ai_detect_sections(pdf_path, total_pages)

    return {
        "MCQ": mcq_pages,
        "OE":  oe_pages,
        "Answers": ans_pages,
        "total": total_pages,
        "mcq_q_map": mcq_q_map,
        "oe_q_map":  oe_q_map,
        "mcq_total": mcq_total,
        "oe_total":  oe_total,
        "is_scanned": not text_found,
    }


def ai_detect_sections(pdf_path: str, total_pages: int) -> dict:
    """Gemini fallback for completely unreadable PDFs."""
    reader = PdfReader(pdf_path)
    sample = ""
    for i in list(range(min(4, total_pages))) + list(range(max(0, total_pages-3), total_pages)):
        sample += f"\n--- Page {i+1} ---\n{(reader.pages[i].extract_text() or '')[:400]}"

    prompt = f"""Singapore primary exam paper, {total_pages} pages total.
Sample text from first and last pages:
{sample}

Identify which pages belong to MCQ, Open-Ended questions, and Answers sections.
Typical structure: cover (1-2 pages), MCQ section (3-6 pages), OE section (5-12 pages), answers (2-5 pages).
Return ONLY JSON (0-indexed page numbers):
{{"MCQ": [2,3,4,5], "OE": [6,7,8,9,10,11], "Answers": [12,13,14],
 "mcq_total": 30, "oe_total": 20, "total": {total_pages},
 "mcq_q_map": {{}}, "oe_q_map": {{}}, "is_scanned": true}}"""

    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30
        )
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            result = json.loads(m.group())
            result["total"] = total_pages
            return result
    except Exception as e:
        logger.error(f"Gemini fallback failed: {e}")

    # Hard fallback: rough split
    cover = 2
    a = cover + int((total_pages - cover) * 0.30)
    b = cover + int((total_pages - cover) * 0.80)
    return {
        "MCQ": list(range(cover, a)),
        "OE":  list(range(a, b)),
        "Answers": list(range(b, total_pages)),
        "total": total_pages,
        "mcq_q_map": {}, "oe_q_map": {},
        "mcq_total": estimate_q_count(a - cover, "MCQ"),
        "oe_total":  estimate_q_count(b - a, "OE"),
        "is_scanned": True,
    }


def get_pages_for_q_range(q_map: dict, all_pages: list, q_from: int, q_to: int, total_q: int) -> list:
    """
    Get pages for a question range.
    Uses exact map if available (text PDF), otherwise estimates from page count (scanned PDF).
    """
    if q_map:
        # Exact lookup
        pages = sorted(set(p for q, p in q_map.items() if q_from <= q <= q_to))
        return pages if pages else all_pages

    if not all_pages:
        return []

    # Scanned PDF: estimate which pages cover q_from..q_to
    q_per_page = max(1, total_q / len(all_pages)) if all_pages else 5
    page_from = max(0, int((q_from - 1) / q_per_page))
    page_to   = min(len(all_pages) - 1, int((q_to - 1) / q_per_page))
    return all_pages[page_from:page_to + 1]


def extract_pages(pdf_path: str, page_indices: list, output_path: str):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for i in page_indices:
        if 0 <= i < len(reader.pages):
            writer.add_page(reader.pages[i])
    with open(output_path, "wb") as f:
        writer.write(f)


def parse_filename(url: str) -> dict:
    filename = url.split("/")[-1].replace(".pdf", "")
    prompt = f"""Parse Singapore exam filename: "{filename}"
Return ONLY JSON: {{"level":"P5","subject":"Science","year":"2022","exam_type":"SA2","school":"ACS"}}"""
    try:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15
        )
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except:
        pass
    parts = filename.split("_")
    return {
        "level":     next((p for p in parts if re.match(r'P\d', p)), "?"),
        "subject":   next((p for p in parts if p in ["Science","Math","English","Chinese","Maths"]), "?"),
        "year":      next((p for p in parts if p.isdigit() and len(p)==4), "?"),
        "exam_type": next((p for p in parts if re.match(r'(WA|SA|CA|Prelim)\d?', p, re.I)), "?"),
        "school":    parts[-1] if parts else "?",
    }


def format_assignees(mentions):
    if not mentions:
        return "Unassigned"
    names = [m["display"] for m in mentions]
    return ", ".join(names[:-1]) + f" & {names[-1]}" if len(names) > 1 else names[0]


def extract_mentions(update: Update):
    mentions = []
    msg = update.message
    if msg.entities:
        for ent in msg.entities:
            if ent.type == "mention":
                uname = msg.text[ent.offset:ent.offset+ent.length].lstrip("@").lower()
                mentions.append({"username": uname, "display": KNOWN_USERS.get(uname, f"@{uname}"), "user_id": None})
            elif ent.type == "text_mention":
                u = ent.user
                uname = (u.username or "").lower()
                mentions.append({"username": uname or str(u.id), "display": KNOWN_USERS.get(uname) or u.full_name, "user_id": u.id})
    if not mentions and msg.text:
        for uname in re.findall(r'@(\w+)', msg.text):
            ul = uname.lower()
            mentions.append({"username": ul, "display": KNOWN_USERS.get(ul, f"@{uname}"), "user_id": None})
    return mentions


# ─── TELEGRAM HANDLERS ───────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📚 *Study Session Archive Bot*\n\n"
        "Paste any PDF link and I'll:\n"
        "• Download + detect sections automatically\n"
        "• Extract exactly the pages you need\n"
        "• Works with scanned exam papers too\n\n"
        "Tag colleagues to track who's printing:\n"
        "`@alice @bob <pdf link>`\n\n"
        "Commands:\n"
        "`/pending @name` — their open jobs\n"
        "`/done @name` — mark printed\n"
        "`/summary` — everyone's status",
        parse_mode="Markdown"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID:
        return

    text = update.message.text or ""
    urls = re.findall(r'https?://\S+\.pdf', text)
    if not urls:
        return

    # Check if waiting for custom range input
    if context.user_data.get("awaiting_custom"):
        await handle_custom_text(update, context)
        return

    url = urls[0]
    mentions = extract_mentions(update)
    assignee_str = format_assignees(mentions)

    msg = await update.message.reply_text(
        f"⏳ Downloading...\n👤 For: *{assignee_str}*", parse_mode="Markdown"
    )

    try:
        r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "/".join(url.split("/")[:3]) + "/"
        }, timeout=30)
        r.raise_for_status()
    except Exception as e:
        await msg.edit_text(f"❌ Download failed: {e}")
        return

    filename = url.split("/")[-1]
    tmp_dir = tempfile.mkdtemp()
    pdf_path = f"{tmp_dir}/{filename}"
    with open(pdf_path, "wb") as f:
        f.write(r.content)

    await msg.edit_text("🔍 Detecting sections...")
    metadata = parse_filename(url)
    metadata["filename"] = filename
    sections = detect_sections(pdf_path)

    context.user_data.update({
        "pdf_path": pdf_path, "sections": sections,
        "metadata": metadata, "filename": filename, "mentions": mentions,
    })

    mcq_total  = sections.get("mcq_total", 0)
    oe_total   = sections.get("oe_total", 0)
    ans_pgs    = len(sections.get("Answers", []))
    is_scanned = sections.get("is_scanned", False)
    scan_note  = " _(estimated — scanned PDF)_" if is_scanned else ""

    mcq_half = max(1, round(mcq_total / 2))
    oe_half  = max(1, round(oe_total / 2))

    assignee_line = f"\n👥 *For:* {assignee_str}" if mentions else ""

    summary = (
        f"✅ *{filename}*\n"
        f"📋 {metadata.get('level')} {metadata.get('subject')} "
        f"| {metadata.get('year')} {metadata.get('exam_type')} — {metadata.get('school')}\n"
        f"📄 {sections['total']} pages total{assignee_line}\n\n"
        f"*Sections detected:*\n"
        f"• MCQ: ~{mcq_total} questions ({len(sections.get('MCQ',[]))} pages){scan_note}\n"
        f"• Open-ended: ~{oe_total} questions ({len(sections.get('OE',[]))} pages){scan_note}\n"
        f"• Answers: {ans_pgs} pages\n\n"
        f"What do you want to extract?"
    )

    keyboard = [
        [
            InlineKeyboardButton(f"📝 First {mcq_half} MCQ",  callback_data=f"ext_mcq_half"),
            InlineKeyboardButton(f"📝 All MCQ",                callback_data="ext_mcq_all"),
        ],
        [
            InlineKeyboardButton(f"📖 First {oe_half} OE",    callback_data=f"ext_oe_half"),
            InlineKeyboardButton(f"📖 All OE",                 callback_data="ext_oe_all"),
        ],
        [
            InlineKeyboardButton("✅ All Answers",             callback_data="ext_answers"),
            InlineKeyboardButton("📄 Questions only",          callback_data="ext_questions"),
        ],
        [
            InlineKeyboardButton("🖨️ All 3 separately",       callback_data="ext_all_three"),
        ],
        [
            InlineKeyboardButton("✏️ Custom range...",         callback_data="ext_custom"),
            InlineKeyboardButton("📁 Save to Drive & Log",     callback_data="save_log"),
        ],
    ]

    await msg.edit_text(summary, parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup(keyboard))


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    action   = query.data
    pdf_path = context.user_data.get("pdf_path")
    sections = context.user_data.get("sections", {})
    filename = context.user_data.get("filename", "paper.pdf")
    mentions = context.user_data.get("mentions", [])

    if not pdf_path or not os.path.exists(pdf_path):
        await query.edit_message_text("❌ Session expired — resend the link.")
        return

    tmp_dir   = os.path.dirname(pdf_path)
    mcq_map   = sections.get("mcq_q_map", {})
    oe_map    = sections.get("oe_q_map", {})
    mcq_total = sections.get("mcq_total", 0)
    oe_total  = sections.get("oe_total", 0)
    mcq_pages = sections.get("MCQ", [])
    oe_pages  = sections.get("OE", [])

    async def send_pdf(pages, label, tag):
        if not pages:
            await query.message.reply_text(f"⚠️ No pages found for: {label}")
            return
        out = f"{tmp_dir}/{filename.replace('.pdf', f'_{tag}.pdf')}"
        extract_pages(pdf_path, pages, out)
        cap = f"📄 *{label}*\n{len(pages)} page(s) · For: {format_assignees(mentions)}"
        with open(out, "rb") as f:
            await context.bot.send_document(
                chat_id=query.message.chat_id,
                document=f, filename=os.path.basename(out),
                caption=cap, parse_mode="Markdown"
            )

    await query.edit_message_text("✂️ Extracting...")

    mcq_half = max(1, round(mcq_total / 2))
    oe_half  = max(1, round(oe_total / 2))

    if action == "ext_mcq_half":
        pages = get_pages_for_q_range(mcq_map, mcq_pages, 1, mcq_half, mcq_total)
        await send_pdf(pages, f"MCQ Q1–{mcq_half}", f"MCQ_Q1-{mcq_half}")

    elif action == "ext_mcq_all":
        await send_pdf(mcq_pages, f"All MCQ (~{mcq_total} questions)", "MCQ_all")

    elif action == "ext_oe_half":
        pages = get_pages_for_q_range(oe_map, oe_pages, 1, oe_half, oe_total)
        await send_pdf(pages, f"OE Q1–{oe_half}", f"OE_Q1-{oe_half}")

    elif action == "ext_oe_all":
        await send_pdf(oe_pages, f"All OE (~{oe_total} questions)", "OE_all")

    elif action == "ext_answers":
        await send_pdf(sections.get("Answers", []), "All Answers", "Answers")

    elif action == "ext_questions":
        await send_pdf(mcq_pages + oe_pages, "Full Questions (no answers)", "Questions_only")

    elif action == "ext_all_three":
        for pages, label, tag in [
            (mcq_pages, f"All MCQ (~{mcq_total}q)", "MCQ_all"),
            (oe_pages,  f"All OE (~{oe_total}q)",   "OE_all"),
            (sections.get("Answers", []), "Answers", "Answers"),
        ]:
            await send_pdf(pages, label, tag)

    elif action == "ext_custom":
        context.user_data["awaiting_custom"] = True
        await query.edit_message_text(
            "✏️ *Custom extraction*\n\n"
            "Type what you want:\n"
            "• `mcq 1-15` — MCQ questions 1 to 15\n"
            "• `oe 5-10` — OE questions 5 to 10\n"
            "• `answers` — answer pages\n"
            "• `pages 3-8` — specific page numbers",
            parse_mode="Markdown"
        )
        return

    elif action == "save_log":
        await query.edit_message_text(
            f"📁 Saved!\n👤 For: {format_assignees(mentions)}\n📊 Check your Google Sheet."
        )
        return

    await query.edit_message_text(
        f"✅ Done!\n{'👤 For: ' + format_assignees(mentions) if mentions else 'Sent to chat.'}"
    )


async def handle_custom_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID:
        return

    text = update.message.text or ""

    if re.search(r'https?://\S+\.pdf', text):
        context.user_data.pop("awaiting_custom", None)
        await handle_message(update, context)
        return

    if not context.user_data.get("awaiting_custom"):
        return

    context.user_data["awaiting_custom"] = False

    sections  = context.user_data.get("sections", {})
    pdf_path  = context.user_data.get("pdf_path")
    filename  = context.user_data.get("filename", "paper.pdf")
    mentions  = context.user_data.get("mentions", [])

    if not pdf_path or not os.path.exists(pdf_path):
        await update.message.reply_text("❌ Session expired — resend the PDF link.")
        return

    text_lower = text.lower().strip()
    tmp_dir    = os.path.dirname(pdf_path)
    mcq_map    = sections.get("mcq_q_map", {})
    oe_map     = sections.get("oe_q_map", {})
    mcq_pages  = sections.get("MCQ", [])
    oe_pages   = sections.get("OE", [])
    mcq_total  = sections.get("mcq_total", 0)
    oe_total   = sections.get("oe_total", 0)

    pages, label = [], text

    mcq_m = re.search(r'mcq\s+(\d+)\s*[-–to]+\s*(\d+)', text_lower)
    oe_m  = re.search(r'oe\s+(\d+)\s*[-–to]+\s*(\d+)',  text_lower)
    pg_m  = re.search(r'pages?\s+(\d+)\s*[-–to]+\s*(\d+)', text_lower)

    if mcq_m:
        q1, q2 = int(mcq_m.group(1)), int(mcq_m.group(2))
        pages = get_pages_for_q_range(mcq_map, mcq_pages, q1, q2, mcq_total)
        label = f"MCQ Q{q1}–{q2}"
    elif oe_m:
        q1, q2 = int(oe_m.group(1)), int(oe_m.group(2))
        pages = get_pages_for_q_range(oe_map, oe_pages, q1, q2, oe_total)
        label = f"OE Q{q1}–{q2}"
    elif pg_m:
        p1, p2 = int(pg_m.group(1)) - 1, int(pg_m.group(2)) - 1
        pages = list(range(p1, p2 + 1))
        label = f"Pages {p1+1}–{p2+1}"
    elif "answer" in text_lower:
        pages = sections.get("Answers", [])
        label = "Answers"
    else:
        await update.message.reply_text(
            "❓ Couldn't parse that. Try:\n`mcq 1-15` · `oe 5-10` · `pages 3-8` · `answers`",
            parse_mode="Markdown"
        )
        return

    if not pages:
        await update.message.reply_text("⚠️ No pages found for that range.")
        return

    out_name = filename.replace(".pdf", f"_{label.replace(' ','_').replace('–','-')}.pdf")
    out_path = f"{tmp_dir}/{out_name}"
    extract_pages(pdf_path, pages, out_path)

    with open(out_path, "rb") as f:
        await update.message.reply_document(
            document=f, filename=out_name,
            caption=f"📄 *{label}* — {len(pages)} page(s) · For: {format_assignees(mentions)}",
            parse_mode="Markdown"
        )


async def cmd_pending(update, context):
    args = context.args
    name = args[0].lstrip("@") if args else "?"
    await update.message.reply_text(f"📋 Pending for @{name}: _(connect Google Sheets to see live data)_", parse_mode="Markdown")

async def cmd_done(update, context):
    args = context.args
    name = args[0].lstrip("@") if args else "?"
    await update.message.reply_text(f"✅ Marked @{name}'s jobs as printed.", parse_mode="Markdown")

async def cmd_summary(update, context):
    await update.message.reply_text("📊 _(connect Google Sheets to see live summary)_", parse_mode="Markdown")


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",   start))
    app.add_handler(CommandHandler("pending", cmd_pending))
    app.add_handler(CommandHandler("done",    cmd_done))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CallbackQueryHandler(handle_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_text))
    logger.info("Archive Bot v4 started!")
    app.run_polling()

if __name__ == "__main__":
    main()
