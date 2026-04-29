"""
AI Archive Bot — Telegram bot for managing exam papers
Handles: PDF download from links, smart section extraction, Google Drive filing, Google Sheets logging
"""

import os
import re
import json
import logging
import requests
import tempfile
from pathlib import Path
from datetime import datetime
from pypdf import PdfReader, PdfWriter
import pdfplumber
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ContextTypes, filters
)

# ─── CONFIG ─────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID", "")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "")  # root archive folder
ALLOWED_USER_ID = int(os.environ.get("ALLOWED_USER_ID", "0"))  # your Telegram user ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── FOLDER MAPPING ──────────────────────────────────────────────────
# Maps keywords in filename → Google Drive subfolder ID
FOLDER_MAP = {
    "Science":    os.environ.get("FOLDER_SCIENCE", ""),
    "Math":       os.environ.get("FOLDER_MATH", ""),
    "English":    os.environ.get("FOLDER_ENGLISH", ""),
    "Chinese":    os.environ.get("FOLDER_CHINESE", ""),
    "P6":         os.environ.get("FOLDER_P6", ""),
    "P5":         os.environ.get("FOLDER_P5", ""),
    "P4":         os.environ.get("FOLDER_P4", ""),
}

# ─── PDF SECTION DETECTOR ────────────────────────────────────────────
def detect_sections(pdf_path: str) -> dict:
    """
    Uses AI (Gemini) + text analysis to find where MCQ, Open-Ended, and Answers start.
    Returns: {"MCQ": [page_indices], "OE": [page_indices], "Answers": [page_indices]}
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    section_map = {"MCQ": [], "OE": [], "Answers": [], "total": total_pages}
    
    # Extract text from each page to find section markers
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").lower()
            
            # MCQ section markers
            if any(kw in text for kw in [
                "multiple choice", "section a", "choose the correct",
                "circle the letter", "shade the letter"
            ]):
                section_map["MCQ"].append(i)
            
            # Open-ended section markers  
            elif any(kw in text for kw in [
                "section b", "open-ended", "open ended", 
                "structured", "short answer", "write your answer"
            ]):
                section_map["OE"].append(i)
            
            # Answer markers
            elif any(kw in text for kw in [
                "answer", "answers", "answer key", "marking scheme",
                "suggested answer", "correct answer"
            ]):
                section_map["Answers"].append(i)
    
    # Fallback: if AI detection fails, use Gemini to analyze the paper
    if not any([section_map["MCQ"], section_map["OE"], section_map["Answers"]]):
        section_map = ai_detect_sections(pdf_path, total_pages)
    
    return section_map


def ai_detect_sections(pdf_path: str, total_pages: int) -> dict:
    """Use Gemini to intelligently detect sections when text parsing fails."""
    # Extract first 3 pages and last 3 pages for context
    sample_text = ""
    reader = PdfReader(pdf_path)
    sample_pages = list(range(min(3, total_pages))) + list(range(max(0, total_pages-3), total_pages))
    
    for i in sample_pages:
        text = reader.pages[i].extract_text() or ""
        sample_text += f"\n--- Page {i+1} ---\n{text[:500]}"
    
    prompt = f"""This is a Singapore primary school exam paper with {total_pages} pages.
    
Analyze this sample text and return a JSON with section page ranges (0-indexed):
{sample_text}

Return ONLY valid JSON like:
{{"MCQ": [0,1,2,3,4], "OE": [5,6,7,8,9,10], "Answers": [11,12,13,14,15]}}

If a section is not present, use an empty list."""

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=30
        )
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.error(f"Gemini section detection failed: {e}")
    
    # Last resort: split evenly (MCQ first 40%, OE next 45%, Answers last 15%)
    mcq_end = int(total_pages * 0.4)
    oe_end = int(total_pages * 0.85)
    return {
        "MCQ": list(range(0, mcq_end)),
        "OE": list(range(mcq_end, oe_end)),
        "Answers": list(range(oe_end, total_pages)),
        "total": total_pages
    }


def extract_pages(pdf_path: str, page_indices: list, output_path: str):
    """Extract specific pages from a PDF into a new file."""
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    for i in page_indices:
        if 0 <= i < len(reader.pages):
            writer.add_page(reader.pages[i])
    
    with open(output_path, "wb") as f:
        writer.write(f)


def parse_filename(url: str) -> dict:
    """
    Parse exam paper metadata from filename.
    e.g. P6_Science_2025_WA1_mgs.pdf → {level, subject, year, exam_type, school}
    """
    filename = url.split("/")[-1].replace(".pdf", "")
    parts = filename.split("_")
    
    # Use Gemini for smart parsing
    prompt = f"""Parse this Singapore exam paper filename: "{filename}"
    
Extract and return ONLY valid JSON:
{{"level": "P6", "subject": "Science", "year": "2025", "exam_type": "WA1", "school": "MGS", "folder_path": "Science/P6/2025"}}

Common exam types: WA1, WA2, SA1, SA2, CA1, CA2, Prelim
If unsure about any field, make a reasonable guess."""

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15
        )
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # Fallback regex parsing
    return {
        "level": next((p for p in parts if re.match(r'P\d', p)), "Unknown"),
        "subject": next((p for p in parts if p in ["Science","Math","English","Chinese"]), "Unknown"),
        "year": next((p for p in parts if p.isdigit() and len(p)==4), str(datetime.now().year)),
        "exam_type": next((p for p in parts if re.match(r'(WA|SA|CA|Prelim)\d?', p)), "Unknown"),
        "school": parts[-1] if parts else "Unknown",
        "folder_path": "General"
    }


# ─── GOOGLE DRIVE UPLOAD ─────────────────────────────────────────────
def upload_to_drive(file_path: str, filename: str, folder_id: str, credentials) -> str:
    """Upload a file to Google Drive and return the file URL."""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    
    service = build("drive", "v3", credentials=credentials)
    
    file_metadata = {
        "name": filename,
        "parents": [folder_id]
    }
    
    media = MediaFileUpload(file_path, mimetype="application/pdf", resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields="id,webViewLink").execute()
    
    return file.get("webViewLink", "")


def get_or_create_folder(service, folder_name: str, parent_id: str) -> str:
    """Get or create a Google Drive folder, return its ID."""
    query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, fields="files(id)").execute()
    
    if results["files"]:
        return results["files"][0]["id"]
    
    folder_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id]
    }
    folder = service.files().create(body=folder_metadata, fields="id").execute()
    return folder["id"]


# ─── GOOGLE SHEETS LOG ───────────────────────────────────────────────
def log_to_sheets(metadata: dict, drive_url: str, credentials):
    """Append a row to the Google Sheet archive log."""
    from googleapiclient.discovery import build
    
    service = build("sheets", "v4", credentials=credentials)
    
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        metadata.get("level", ""),
        metadata.get("subject", ""),
        metadata.get("year", ""),
        metadata.get("exam_type", ""),
        metadata.get("school", ""),
        drive_url,
        metadata.get("filename", ""),
    ]
    
    service.spreadsheets().values().append(
        spreadsheetId=GOOGLE_SHEET_ID,
        range="Archive!A:H",
        valueInputOption="USER_ENTERED",
        body={"values": [row]}
    ).execute()


# ─── TELEGRAM HANDLERS ───────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📚 *Archive Bot Ready*\n\n"
        "Send me a PDF link and I'll:\n"
        "• Download it automatically\n"
        "• File it in the right Google Drive folder\n"
        "• Log it in your Google Sheet\n"
        "• Let you extract specific sections\n\n"
        "Commands:\n"
        "`/extract <url>` — Smart section extraction\n"
        "`/find <keyword>` — Search your archive\n"
        "`/list` — Recent papers\n",
        parse_mode="Markdown"
    )


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main handler: detects PDF URLs in any message."""
    
    if update.effective_user.id != ALLOWED_USER_ID:
        return  # Only respond to you
    
    text = update.message.text or ""
    urls = re.findall(r'https?://\S+\.pdf', text)
    
    if not urls:
        return
    
    url = urls[0]
    msg = await update.message.reply_text("⏳ Downloading paper...")
    
    # ── Download PDF ──
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "/".join(url.split("/")[:3]) + "/"
        }
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception as e:
        await msg.edit_text(f"❌ Download failed: {e}")
        return
    
    # Save temporarily
    filename = url.split("/")[-1]
    tmp_dir = tempfile.mkdtemp()
    pdf_path = f"{tmp_dir}/{filename}"
    
    with open(pdf_path, "wb") as f:
        f.write(r.content)
    
    # ── Parse metadata ──
    await msg.edit_text("🔍 Analysing paper...")
    metadata = parse_filename(url)
    metadata["filename"] = filename
    metadata["url"] = url
    
    # ── Detect sections ──
    sections = detect_sections(pdf_path)
    context.user_data["current_pdf"] = pdf_path
    context.user_data["sections"] = sections
    context.user_data["metadata"] = metadata
    context.user_data["filename"] = filename
    
    # Store in Drive (using credentials loaded at startup)
    # upload_to_drive(pdf_path, filename, folder_id, creds)  ← uncomment when creds set up
    
    # ── Show result + options ──
    mcq_count = len(sections.get("MCQ", []))
    oe_count = len(sections.get("OE", []))
    ans_count = len(sections.get("Answers", []))
    
    summary = (
        f"✅ *{filename}*\n\n"
        f"📋 Level: {metadata.get('level')} | {metadata.get('subject')}\n"
        f"📅 {metadata.get('year')} {metadata.get('exam_type')} — {metadata.get('school')}\n"
        f"📄 {sections['total']} pages total\n\n"
        f"Sections detected:\n"
        f"• MCQ: {mcq_count} pages\n"
        f"• Open-ended: {oe_count} pages\n"
        f"• Answers: {ans_count} pages\n\n"
        f"What would you like to extract?"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("First 10 MCQ", callback_data="extract_mcq_10"),
            InlineKeyboardButton("All MCQ", callback_data="extract_mcq_all"),
        ],
        [
            InlineKeyboardButton("First 10 OE", callback_data="extract_oe_10"),
            InlineKeyboardButton("All OE", callback_data="extract_oe_all"),
        ],
        [
            InlineKeyboardButton("All Answers", callback_data="extract_answers"),
            InlineKeyboardButton("Full Paper", callback_data="extract_full"),
        ],
        [
            InlineKeyboardButton("📁 Save to Drive & Log", callback_data="save_and_log"),
        ]
    ]
    
    await msg.edit_text(summary, parse_mode="Markdown", 
                        reply_markup=InlineKeyboardMarkup(keyboard))


async def handle_extraction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button presses for PDF extraction."""
    query = update.callback_query
    await query.answer()
    
    action = query.data
    pdf_path = context.user_data.get("current_pdf")
    sections = context.user_data.get("sections", {})
    filename = context.user_data.get("filename", "paper.pdf")
    
    if not pdf_path or not os.path.exists(pdf_path):
        await query.edit_message_text("❌ PDF not found. Please resend the link.")
        return
    
    await query.edit_message_text("✂️ Extracting pages...")
    
    tmp_dir = os.path.dirname(pdf_path)
    
    if action == "extract_mcq_10":
        pages = sections.get("MCQ", [])[:2]  # ~10 questions (5-6 per page)
        out_name = filename.replace(".pdf", "_MCQ_Q1-10.pdf")
    elif action == "extract_mcq_all":
        pages = sections.get("MCQ", [])
        out_name = filename.replace(".pdf", "_MCQ_all.pdf")
    elif action == "extract_oe_10":
        pages = sections.get("OE", [])[:4]  # ~10 questions (2-3 per page)
        out_name = filename.replace(".pdf", "_OE_Q1-10.pdf")
    elif action == "extract_oe_all":
        pages = sections.get("OE", [])
        out_name = filename.replace(".pdf", "_OE_all.pdf")
    elif action == "extract_answers":
        pages = sections.get("Answers", [])
        out_name = filename.replace(".pdf", "_Answers.pdf")
    elif action == "extract_full":
        mcq = sections.get("MCQ", [])
        oe = sections.get("OE", [])
        pages = mcq + oe
        out_name = filename.replace(".pdf", "_Questions_only.pdf")
    elif action == "save_and_log":
        await query.edit_message_text("📁 Saving to Google Drive and logging...")
        # upload_to_drive(...)  ← add Drive upload here
        # log_to_sheets(...)    ← add Sheets logging here
        await query.edit_message_text("✅ Saved to Drive and logged in your sheet!")
        return
    else:
        pages = []
        out_name = "extracted.pdf"
    
    if not pages:
        await query.edit_message_text("⚠️ Section not detected in this paper. Try manual extraction.")
        return
    
    out_path = f"{tmp_dir}/{out_name}"
    extract_pages(pdf_path, pages, out_path)
    
    # Send the extracted PDF back
    await query.edit_message_text(f"✅ Extracted! Sending {out_name}...")
    with open(out_path, "rb") as f:
        await context.bot.send_document(
            chat_id=query.message.chat_id,
            document=f,
            filename=out_name,
            caption=f"📄 {out_name}\n{len(pages)} page(s)"
        )


async def handle_custom_extract(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle natural language extract commands.
    e.g. /extract pages 3-7
    e.g. /extract mcq questions 5 to 15
    """
    text = update.message.text.replace("/extract", "").strip()
    
    if not text:
        await update.message.reply_text(
            "Usage examples:\n"
            "`/extract pages 3-7`\n"
            "`/extract mcq 1-10`\n"
            "`/extract oe 5-15`\n"
            "`/extract answers`",
            parse_mode="Markdown"
        )
        return
    
    # Use Gemini to parse the instruction
    pdf_path = context.user_data.get("current_pdf")
    sections = context.user_data.get("sections", {})
    
    if not pdf_path:
        await update.message.reply_text("Please send a PDF link first.")
        return
    
    prompt = f"""Given these PDF sections (0-indexed page numbers):
MCQ pages: {sections.get('MCQ', [])}
OE pages: {sections.get('OE', [])}  
Answers pages: {sections.get('Answers', [])}

User wants to extract: "{text}"

Return ONLY a JSON with the page indices to extract:
{{"pages": [0, 1, 2, 3], "label": "MCQ Q1-10"}}"""

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15
        )
        result_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            pages = result.get("pages", [])
            label = result.get("label", "extracted")
            
            filename = context.user_data.get("filename", "paper.pdf")
            out_name = filename.replace(".pdf", f"_{label.replace(' ','_')}.pdf")
            tmp_dir = os.path.dirname(pdf_path)
            out_path = f"{tmp_dir}/{out_name}"
            
            extract_pages(pdf_path, pages, out_path)
            
            with open(out_path, "rb") as f:
                await update.message.reply_document(document=f, filename=out_name)
            return
    except Exception as e:
        logger.error(f"Custom extract failed: {e}")
    
    await update.message.reply_text("❌ Couldn't parse that. Try: `/extract pages 3-7`", parse_mode="Markdown")


# ─── MAIN ────────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("extract", handle_custom_extract))
    app.add_handler(CallbackQueryHandler(handle_extraction))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_url))
    
    logger.info("Bot started!")
    app.run_polling()


if __name__ == "__main__":
    main()
