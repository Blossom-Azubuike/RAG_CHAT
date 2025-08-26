import os
import re
import yt_dlp
from langchain_core.documents import Document
from typing import List

def _parse_transcript(raw_text: str) -> str:
    text = re.sub(r'<[^>]+>', '', raw_text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'^Kind:.*?\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^WEBVTT.*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*\n','',text)
    lines = text.strip().split('\n')
    cleaned_lines = []
    for line in lines:
        clean_line = line.strip().lstrip('> ').strip()
        if clean_line and (not cleaned_lines or cleaned_lines[-1] != clean_line):
            cleaned_lines.append(clean_line)
    return " ".join(cleaned_lines).strip()

def load_from_youtube(youtube_url: str) -> List[Document]:
    temp_filename_template = "temp_transcript_%(id)s"
    ydl_opts = {
        'format': 'best',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'skip_download': True,
        'outtmpl': temp_filename_template,
        'quiet': True,
        'ignoreerrors': True,
    }

    transcript_text = ""
    downloaded_file = None
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            video_id = info_dict.get('id', 'default')
            expected_filename = f"temp_transcript_{video_id}.en.vtt"
            ydl.download([youtube_url])

            if os.path.exists(expected_filename):
                downloaded_file = expected_filename
                with open(downloaded_file, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                transcript_text = _parse_transcript(raw_content)
            else:
                transcript_text = ""

    except Exception:
        transcript_text = ""

    finally:
        if downloaded_file and os.path.exists(downloaded_file):
            os.remove(downloaded_file)

    if not transcript_text:
        transcript_text = "No transcript available for this video."

    doc = Document(
        page_content=transcript_text,
        metadata={"source": youtube_url}
    )

    return [doc]
