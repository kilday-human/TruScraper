#!/usr/bin/env python3
"""
TruScraper v2 - Using yt-dlp for more reliable transcript extraction
"""

import subprocess
import json
import sys
import re
import os

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

def check_yt_dlp():
    """Check if yt-dlp is installed"""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_video_info(video_url):
    """Get video info including available subtitles"""
    try:
        cmd = [
            'yt-dlp',
            '--dump-json',
            '--list-subs',
            video_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        video_info = json.loads(result.stdout)
        
        return video_info
    except subprocess.CalledProcessError as e:
        print(f"❌ Error getting video info: {e.stderr}")
        return None
    except json.JSONDecodeError:
        print("❌ Error parsing video information")
        return None

def download_subtitles(video_url, video_id):
    """Download subtitles using yt-dlp"""
    try:
        # Try to download auto-generated English subtitles
        cmd = [
            'yt-dlp',
            '--write-auto-subs',
            '--write-subs',
            '--sub-langs', 'en',
            '--sub-format', 'vtt',
            '--skip-download',
            '--output', f'transcript_{video_id}.%(ext)s',
            video_url
        ]
        
        print("📥 Downloading subtitles with yt-dlp...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Look for generated subtitle files
            possible_files = [
                f'transcript_{video_id}.en.vtt',
                f'transcript_{video_id}.en-US.vtt',
                f'transcript_{video_id}.en-GB.vtt'
            ]
            
            for filename in possible_files:
                if os.path.exists(filename):
                    print(f"✅ Downloaded subtitles: {filename}")
                    return filename
            
            print("⚠️  yt-dlp ran successfully but no subtitle file found")
            return None
        else:
            print(f"❌ yt-dlp error: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading subtitles: {e}")
        return None

def parse_vtt_file(vtt_file):
    """Parse VTT subtitle file"""
    try:
        with open(vtt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove VTT header and metadata
        lines = content.split('\n')
        transcript_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and VTT metadata
            if not line or line.startswith('WEBVTT') or line.startswith('NOTE'):
                i += 1
                continue
            
            # Look for timestamp lines (format: 00:00:00.000 --> 00:00:05.000)
            if '-->' in line:
                timestamp = line.split('-->')[0].strip()
                i += 1
                
                # Get the subtitle text (may be multiple lines)
                subtitle_text = []
                while i < len(lines) and lines[i].strip():
                    subtitle_text.append(lines[i].strip())
                    i += 1
                
                if subtitle_text:
                    # Clean up the text (remove HTML tags)
                    text = ' '.join(subtitle_text)
                    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
                    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                    
                    transcript_lines.append({
                        'timestamp': timestamp,
                        'text': text
                    })
            else:
                i += 1
        
        return transcript_lines
        
    except Exception as e:
        print(f"❌ Error parsing VTT file: {e}")
        return None

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 scraper_v2.py <youtube_url>")
        print("Example: python3 scraper_v2.py https://www.youtube.com/watch?v=mxYtd9DVIfU")
        sys.exit(1)
    
    youtube_url = sys.argv[1]
    
    # Check if yt-dlp is installed
    if not check_yt_dlp():
        print("❌ yt-dlp is not installed!")
        print("Install it with: pip install yt-dlp")
        sys.exit(1)
    
    try:
        # Extract video ID
        video_id = extract_video_id(youtube_url)
        print(f"📹 Video ID: {video_id}")
        
        # Get video info
        print("🔍 Getting video information...")
        video_info = get_video_info(youtube_url)
        
        if video_info:
            print(f"📺 Title: {video_info.get('title', 'Unknown')}")
            
            # Check available subtitles
            subs = video_info.get('subtitles', {})
            auto_subs = video_info.get('automatic_captions', {})
            
            print(f"📝 Manual subtitles: {list(subs.keys())}")
            print(f"🤖 Auto-generated: {list(auto_subs.keys())}")
        
        # Download subtitles
        subtitle_file = download_subtitles(youtube_url, video_id)
        
        if subtitle_file:
            # Parse the VTT file
            transcript_data = parse_vtt_file(subtitle_file)
            
            if transcript_data:
                print(f"\n📄 Transcript ({len(transcript_data)} segments):")
                print("=" * 50)
                
                # Display transcript
                for entry in transcript_data:
                    print(f"[{entry['timestamp']}] {entry['text']}")
                
                # Save clean text version
                text_file = f"transcript_{video_id}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    for entry in transcript_data:
                        f.write(f"[{entry['timestamp']}] {entry['text']}\n")
                
                print(f"\n💾 Clean transcript saved to: {text_file}")
                print(f"🗂️  Raw VTT file: {subtitle_file}")
                
            else:
                print("❌ Could not parse subtitle file")
        else:
            print("❌ Could not download subtitles")
            
    except ValueError as e:
        print(f"❌ URL Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
