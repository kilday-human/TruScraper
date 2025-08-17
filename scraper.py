#!/usr/bin/env python3
"""
TruScraper - YouTube Transcript Extractor
Handles both manual and auto-generated captions
"""

import re
import sys
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise ValueError(f"Could not extract video ID from URL: {url}")

def get_transcript(video_id, language_codes=['en']):
    """
    Get transcript for a video, trying manual first, then auto-generated
    
    Args:
        video_id (str): YouTube video ID
        language_codes (list): List of language codes to try, default ['en']
    
    Returns:
        list: Transcript entries with 'text', 'start', and 'duration' keys
    """
    try:
        # Get all available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # First try to get manual/human transcripts
        try:
            transcript = transcript_list.find_transcript(language_codes)
            print(f"✅ Found manual transcript in: {transcript.language}")
            return transcript.fetch()
        except NoTranscriptFound:
            print("⚠️  No manual transcript found, trying auto-generated...")
        
        # Fall back to auto-generated transcripts
        try:
            transcript = transcript_list.find_generated_transcript(language_codes)
            print(f"✅ Found auto-generated transcript in: {transcript.language}")
            return transcript.fetch()
        except NoTranscriptFound:
            print("❌ No auto-generated transcript found either")
            return None
            
    except TranscriptsDisabled:
        print("❌ Transcripts are disabled for this video")
        return None
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        return None

def format_transcript(transcript_data, include_timestamps=True):
    """
    Format transcript data into readable text
    
    Args:
        transcript_data (list): Raw transcript data
        include_timestamps (bool): Whether to include timestamps
    
    Returns:
        str: Formatted transcript text
    """
    if not transcript_data:
        return "No transcript available"
    
    formatted_lines = []
    
    for entry in transcript_data:
        text = entry['text'].strip()
        if not text:
            continue
            
        if include_timestamps:
            # Convert seconds to MM:SS format
            start_time = entry['start']
            minutes = int(start_time // 60)
            seconds = int(start_time % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            formatted_lines.append(f"{timestamp} {text}")
        else:
            formatted_lines.append(text)
    
    return '\n'.join(formatted_lines)

def main():
    """Main function to run the scraper"""
    if len(sys.argv) != 2:
        print("Usage: python scraper.py <youtube_url>")
        print("Example: python scraper.py https://www.youtube.com/watch?v=mxYtd9DVIfU")
        sys.exit(1)
    
    youtube_url = sys.argv[1]
    
    try:
        # Extract video ID
        video_id = extract_video_id(youtube_url)
        print(f"📹 Video ID: {video_id}")
        
        # Get transcript
        transcript_data = get_transcript(video_id)
        
        if transcript_data:
            print(f"\n📄 Transcript ({len(transcript_data)} segments):")
            print("=" * 50)
            
            # Print formatted transcript
            formatted_text = format_transcript(transcript_data, include_timestamps=True)
            print(formatted_text)
            
            # Also save to file
            filename = f"transcript_{video_id}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            print(f"\n💾 Transcript saved to: {filename}")
            
        else:
            print("❌ Could not retrieve transcript for this video")
            sys.exit(1)
            
    except ValueError as e:
        print(f"❌ URL Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
