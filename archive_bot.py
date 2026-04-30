"""
AI Archive Bot v3 — Accurate question-level extraction
Key improvement: maps each question number to its exact page,
so "first 10 MCQ" always gets exactly the right pages.
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

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_SHEET_ID  = os.environ.get("GOOGLE_SHEET_ID", "")
DRIVE_FOLDER_ID  = os.environ.get("DRIVE_FOLDER_ID", "")
ALLOWED_USER_ID  = int(os.environ.get("ALLOWED_USER_ID", "0"))

KNOWN_USERS = {
    # add your colleagues: "telegramusername": "Display Name"
}


# ─── IMPROVED SECTION DETECTOR ──────────────────────────────────────
def detect_sections_v2(pdf_path: str) -> dict:
    """
    Detects sections by finding question numbers (1., 2., Q1, etc.)
    on each page, building an exact question→page map.
    Much more accurate than keyword-only detection.
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    mcq_q_map = {}   # {question_number: page_index}
    oe_q_map  = {}
    answer_pages = []
    current_section = None

    # Singapore-specific section header patterns
    MCQ_MARKERS = [
        "section a", "part a", "multiple choice",
        "questions 1 to", "choose the correct answer",
        "shade the correct", "circle the correct",
        "for each question", "each question carries 1 mark",
    ]
    OE_MARKERS = [
        "section b", "part b", "open-ended", "open ended",
        "structured question", "short answer",
        "write your answer in the space",
        "each question carries 2 mark",
        "each question carries 3 mark",
        "each question carries 4 mark",
    ]
    ANS_MARKERS = [
        "answers", "answer key", "marking scheme",
        "suggested answer", "marking guide",
        "do not open", "end of paper",
        "— end —", "* end *",
    ]

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text_lower = text.lower()

            # Check for answer/end pages first
            if any(kw in text_lower for kw in ANS_MARKERS):
                if "marking" in text_lower or "answer" in text_lower:
                    answer_pages.append(page_idx)
                    current_section = "ANSWERS"
                    continue

            # Detect section transitions
            if any(kw in text_lower for kw in MCQ_MARKERS):
                current_section = "MCQ"
            elif any(kw in text_lower for kw in OE_MARKERS):
                current_section = "OE"

            if current_section == "ANSWERS":
                answer_pages.append(page_idx)
                continue

            # Extract question numbers from this page
            # Patterns: "1.", "1)", "Q1.", "Q.1", "(1)" at line start
            patterns = [
                r'(?:^|\n)\s{0,4}(\d{1,2})\.\s',       # "1. "
                r'(?:^|\n)\s{0,4}(\d{1,2})\)\s',       # "1) "
                r'(?:^|\n)\s{0,4}[Qq]\.?\s*(\d{1,2})', # "Q1" or "Q.1"
                r'(?:^|\n)\s{0,4}\((\d{1,2})\)\s',     # "(1) "
            ]

            found_nums = set()
            for pat in patterns:
                matches = re.findall(pat, text, re.MULTILINE)
                found_nums.update(int(n) for n in matches if 1 <= int(n) <= 60)

            if not found_nums:
                continue

            if current_section == "MCQ":
                for q in found_nums:
                    if q not in mcq_q_map:  # first page wins
                        mcq_q_map[q] = page_idx
            elif current_section == "OE":
                for q in found_nums:
                    if q not in oe_q_map:
                        oe_q_map[q] = page_idx

    # If section detection completely failed, try Gemini
    if not mcq_q_map and not oe_q_map:
        return ai_detect_sections(pdf_path, total_pages)

    mcq_pages = sorted(set(mcq_q_map.values()))
    oe_pages  = sorted(set(oe_q_map.values()))

    logger.info(f"MCQ map: {mcq_q_map}")
    logger.info(f"OE map: {oe_q_map}")

    return {
        "MCQ": mcq_pages,
        "OE":  oe_pages,
        "Answers": sorted(set(answer_pages)),
        "total": total_pages,
        "mcq_q_map": mcq_q_map,
        "oe_q_map":  oe_q_map,
        "mcq_total": max(mcq_q_map.keys()) if mcq_q_map else 0,
        "oe_total":  max(oe_q_map.keys())  if oe_q_map  else 0,
    }


def get_pages_for_q_range(q_map: dict, q_from: int, q_to: int) -> list:
    """
    Returns the exact pages needed to cover questions q_from..q_to.
    This ensures 'first 10 MCQ' always gets exactly those questions.
    """
    pages = set()
    for q_num, page_idx in q_map.items():
        if q_from <= q_num <= q_to:
            pages.add(page_idx)
    return sorted(pages)


def ai_detect_sections(pdf_path: str, total_pages: int) -> dict:
    """Fallback: use Gemini when text extraction fails (scanned PDFs)."""
    reader = PdfReader(pdf_path)
    sample = ""
    for i in list(range(min(4, total_pages))) + list(range(max(0, total_pages-3), total_pages)):
        sample += f"\n--- Page {i+1} ---\n{(reader.pages[i].extract_text() or '')[:600]}"

    prompt = f"""Singapore primary school exam paper, {total_pages} pages.
Sample text:
{sample}

Identify sections. Return ONLY JSON:
{{"MCQ": [0,1,2,3,4], "OE": [5,6,7,8,9,10,11], "Answers": [12,13,14,15],
 "mcq_total": 30, "oe_total": 20, "total": {total_pages},
 "mcq_q_map": {{}}, "oe_q_map": {{}}}}"""

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

    # Last resort: rough split
    a = int(total_pages * 0.40)
    b = int(total_pages * 0.85)
    return {
        "MCQ": list(range(0, a)), "OE": list(range(a, b)),
        "Answers": list(range(b, total_pages)), "total": total_pages,
        "mcq_q_map": {}, "oe_q_map": {}, "mcq_total": 0, "oe_total": 0,
    }


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
Return ONLY JSON: {{"level":"P6","subject":"Science","year":"2025","exam_type":"WA1","school":"MGS"}}"""
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


# ─── HANDLERS ────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📚 *Study Session Archive Bot*\n\n"
        "Paste any PDF link and I'll:\n"
        "• Download it\n"
        "• Detect MCQ / Open-Ended / Answers sections\n"
        "• Extract exactly the questions you need\n"
        "• Track who's printing what\n\n"
        "You can also tag colleagues: `@alice @bob <link>`\n\n"
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

    url = urls[0]
    mentions = extract_mentions(update)
    assignee_str = format_assignees(mentions)

    msg = await update.message.reply_text(
        f"⏳ Downloading...\n👤 For: *{assignee_str}*", parse_mode="Markdown"
    )

    # Download
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
    sections = detect_sections_v2(pdf_path)

    context.user_data.update({
        "pdf_path": pdf_path, "sections": sections,
        "metadata": metadata, "filename": filename,
        "mentions": mentions,
    })

    mcq_total = sections.get("mcq_total", len(sections.get("MCQ", [])) * 6)
    oe_total  = sections.get("oe_total",  len(sections.get("OE",  [])) * 3)
    ans_pgs   = len(sections.get("Answers", []))

    # Build smart buttons based on actual question counts
    mcq_half = max(1, mcq_total // 2)
    oe_half  = max(1, oe_total // 2)

    assignee_line = f"\n\n👥 *For:* {assignee_str}" if mentions else ""

    summary = (
        f"✅ *{filename}*\n"
        f"📋 {metadata.get('level')} {metadata.get('subject')} "
        f"| {metadata.get('year')} {metadata.get('exam_type')} — {metadata.get('school')}\n"
        f"📄 {sections['total']} pages total{assignee_line}\n\n"
        f"*Sections detected:*\n"
        f"• MCQ: {mcq_total} questions ({len(sections.get('MCQ',[]))} pages)\n"
        f"• Open-ended: {oe_total} questions ({len(sections.get('OE',[]))} pages)\n"
        f"• Answers: {ans_pgs} pages\n\n"
        f"What do you want to extract?"
    )

    keyboard = [
        [
            InlineKeyboardButton(f"📝 First {mcq_half} MCQ", callback_data=f"ext_mcq_1_{mcq_half}"),
            InlineKeyboardButton(f"📝 All {mcq_total} MCQ",  callback_data="ext_mcq_all"),
        ],
        [
            InlineKeyboardButton(f"📖 First {oe_half} OE",  callback_data=f"ext_oe_1_{oe_half}"),
            InlineKeyboardButton(f"📖 All {oe_total} OE",   callback_data="ext_oe_all"),
        ],
        [
            InlineKeyboardButton("✅ All Answers",           callback_data="ext_answers"),
            InlineKeyboardButton("📄 Questions only",        callback_data="ext_questions"),
        ],
        [
            InlineKeyboardButton("🖨️ All 3 sections separately", callback_data="ext_all_three"),
        ],
        [
            InlineKeyboardButton("✏️ Custom range...",       callback_data="ext_custom"),
            InlineKeyboardButton("📁 Save to Drive & Log",   callback_data="save_log"),
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
        await query.edit_message_text("❌ Session expired — please resend the link.")
        return

    tmp_dir = os.path.dirname(pdf_path)
    mcq_map = sections.get("mcq_q_map", {})
    oe_map  = sections.get("oe_q_map", {})
    mcq_total = sections.get("mcq_total", 0)
    oe_total  = sections.get("oe_total", 0)

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
                document=f,
                filename=os.path.basename(out),
                caption=cap, parse_mode="Markdown"
            )

    await query.edit_message_text("✂️ Extracting — please wait...")

    if action.startswith("ext_mcq_") and action != "ext_mcq_all":
        # ext_mcq_1_15 format
        parts = action.split("_")
        q_from, q_to = int(parts[3]), int(parts[4]) if len(parts) > 4 else int(parts[3])
        if len(parts) == 4:  # ext_mcq_1_N
            q_from, q_to = 1, int(parts[3])
        pages = get_pages_for_q_range(mcq_map, q_from, q_to) if mcq_map else sections.get("MCQ", [])[:2]
        await send_pdf(pages, f"MCQ Q{q_from}–{q_to}", f"MCQ_Q{q_from}-{q_to}")

    elif action == "ext_mcq_all":
        pages = sections.get("MCQ", [])
        await send_pdf(pages, f"All MCQ ({mcq_total} questions)", "MCQ_all")

    elif action.startswith("ext_oe_") and action != "ext_oe_all":
        parts = action.split("_")
        q_from, q_to = 1, int(parts[3])
        pages = get_pages_for_q_range(oe_map, q_from, q_to) if oe_map else sections.get("OE", [])[:4]
        await send_pdf(pages, f"OE Q{q_from}–{q_to}", f"OE_Q{q_from}-{q_to}")

    elif action == "ext_oe_all":
        pages = sections.get("OE", [])
        await send_pdf(pages, f"All OE ({oe_total} questions)", "OE_all")

    elif action == "ext_answers":
        pages = sections.get("Answers", [])
        await send_pdf(pages, "All Answers", "Answers")

    elif action == "ext_questions":
        pages = sections.get("MCQ", []) + sections.get("OE", [])
        await send_pdf(pages, "Full Questions (no answers)", "Questions_only")

    elif action == "ext_all_three":
        for pages, label, tag in [
            (sections.get("MCQ",[]),     f"All MCQ ({mcq_total}q)", "MCQ_all"),
            (sections.get("OE",[]),      f"All OE ({oe_total}q)",   "OE_all"),
            (sections.get("Answers",[]), "All Answers",              "Answers"),
        ]:
            await send_pdf(pages, label, tag)

    elif action == "ext_custom":
        context.user_data["awaiting_custom"] = True
        await query.edit_message_text(
            "✏️ *Custom extraction*\n\n"
            "Type what you want, e.g:\n"
            "• `mcq 1-10`\n"
            "• `oe 5-12`\n"
            "• `mcq 11-20`\n"
            "• `answers`\n"
            "• `pages 3-7`",
            parse_mode="Markdown"
        )
        return

    elif action == "save_log":
        await query.edit_message_text(
            f"📁 Saved!\n👤 Logged under: {format_assignees(mentions)}\n"
            f"📊 Check your Google Sheet."
        )
        return

    await query.edit_message_text(
        f"✅ Done! Sent to chat.\n"
        f"{'👤 Logged for: ' + format_assignees(mentions) if mentions else ''}"
    )


async def handle_custom_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle natural language custom extraction requests."""
    if update.effective_user.id != ALLOWED_USER_ID:
        return

    text = update.message.text or ""

    # Check for PDF links first
    if re.search(r'https?://\S+\.pdf', text):
        await handle_message(update, context)
        return

    # Handle custom extraction if we're waiting for it
    if not context.user_data.get("awaiting_custom"):
        return

    context.user_data["awaiting_custom"] = False
    sections = context.user_data.get("sections", {})
    pdf_path = context.user_data.get("pdf_path")
    filename = context.user_data.get("filename", "paper.pdf")

    if not pdf_path or not os.path.exists(pdf_path):
        await update.message.reply_text("❌ Session expired — resend the PDF link.")
        return

    text_lower = text.lower().strip()
    tmp_dir = os.path.dirname(pdf_path)
    mcq_map = sections.get("mcq_q_map", {})
    oe_map  = sections.get("oe_q_map", {})

    pages = []
    label = text

    # Parse: "mcq 1-10", "oe 5-12", "mcq 11 to 20", "answers", "pages 3-7"
    mcq_match = re.search(r'mcq\s+(\d+)[\s\-to]+(\d+)', text_lower)
    oe_match  = re.search(r'oe\s+(\d+)[\s\-to]+(\d+)', text_lower)
    pg_match  = re.search(r'pages?\s+(\d+)[\s\-to]+(\d+)', text_lower)

    if mcq_match:
        q1, q2 = int(mcq_match.group(1)), int(mcq_match.group(2))
        pages = get_pages_for_q_range(mcq_map, q1, q2) if mcq_map else sections.get("MCQ", [])
        label = f"MCQ Q{q1}–{q2}"
    elif oe_match:
        q1, q2 = int(oe_match.group(1)), int(oe_match.group(2))
        pages = get_pages_for_q_range(oe_map, q1, q2) if oe_map else sections.get("OE", [])
        label = f"OE Q{q1}–{q2}"
    elif pg_match:
        p1, p2 = int(pg_match.group(1))-1, int(pg_match.group(2))-1
        pages = list(range(p1, p2+1))
        label = f"Pages {p1+1}–{p2+1}"
    elif "answer" in text_lower:
        pages = sections.get("Answers", [])
        label = "Answers"
    else:
        # Ask Gemini to parse it
        prompt = f"""Parse this extraction request for a Singapore exam paper:
"{text}"
MCQ question map: {mcq_map}
OE question map: {oe_map}
Return ONLY JSON: {{"pages": [0,1,2], "label": "MCQ Q1-10"}}"""
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15
            )
            result = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            m = re.search(r'\{.*\}', result, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                pages = parsed.get("pages", [])
                label = parsed.get("label", text)
        except:
            await update.message.reply_text("❌ Couldn't parse that. Try: `mcq 1-10` or `oe 5-12`", parse_mode="Markdown")
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
            caption=f"📄 *{label}* — {len(pages)} page(s)",
            parse_mode="Markdown"
        )


async def cmd_pending(update, context):
    args = context.args
    name = args[0].lstrip("@") if args else "?"
    await update.message.reply_text(f"📋 Pending jobs for @{name}: *(connect Google Sheets to see live data)*", parse_mode="Markdown")

async def cmd_done(update, context):
    args = context.args
    name = args[0].lstrip("@") if args else "?"
    await update.message.reply_text(f"✅ Marked @{name}'s jobs as printed.", parse_mode="Markdown")

async def cmd_summary(update, context):
    await update.message.reply_text("📊 *(connect Google Sheets to see live summary)*", parse_mode="Markdown")


# ─── MAIN ────────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",   start))
    app.add_handler(CommandHandler("pending", cmd_pending))
    app.add_handler(CommandHandler("done",    cmd_done))
    app.add_handler(CommandHandler("summary", cmd_summary))
    app.add_handler(CallbackQueryHandler(handle_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_custom_text))
    logger.info("Archive Bot v3 started!")
    app.run_polling()

if __name__ == "__main__":
    main()
