#!/usr/bin/env python3
"""
Debug script to see what's happening with the transcript
"""

from youtube_transcript_api import YouTubeTranscriptApi
import json

video_id = "mxYtd9DVIfU"

print(f"🔍 Debugging video: {video_id}")
print("=" * 50)

try:
    # Step 1: List all available transcripts
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    print("📋 Available transcripts:")
    
    for transcript in transcript_list:
        print(f"  - {transcript.language} ({'auto-generated' if transcript.is_generated else 'manual'})")
    
    print("\n" + "=" * 50)
    
    # Step 2: Try to get the English transcript
    print("🎯 Attempting to fetch English transcript...")
    
    # Try the direct approach first
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        print(f"✅ SUCCESS: Got {len(transcript_data)} transcript entries")
        
        # Show first few entries
        print("\n📄 First 3 transcript entries:")
        for i, entry in enumerate(transcript_data[:3]):
            print(f"  {i+1}. [{entry['start']:.1f}s] {entry['text'][:50]}...")
            
    except Exception as e:
        print(f"❌ Direct method failed: {e}")
        
        # Try the manual method
        print("\n🔄 Trying manual transcript selection...")
        try:
            transcript = transcript_list.find_transcript(['en'])
            print(f"Found transcript: {transcript.language}")
            
            # Try to fetch it
            transcript_data = transcript.fetch()
            print(f"✅ SUCCESS: Got {len(transcript_data)} transcript entries")
            
        except Exception as e2:
            print(f"❌ Manual method also failed: {e2}")
            
            # Try generated transcript
            print("\n🤖 Trying auto-generated transcript...")
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
                print(f"Found generated transcript: {transcript.language}")
                
                transcript_data = transcript.fetch()
                print(f"✅ SUCCESS: Got {len(transcript_data)} transcript entries")
                
            except Exception as e3:
                print(f"❌ Generated transcript failed too: {e3}")
                print("\n🚨 All methods failed!")

except Exception as main_error:
    print(f"💥 Main error: {main_error}")
    
print("\n" + "=" * 50)
print("Debug complete!")
