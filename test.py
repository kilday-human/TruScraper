#!/usr/bin/env python3
"""
Test with a known working video to see if the API works at all
"""

from youtube_transcript_api import YouTubeTranscriptApi

# Test with a popular TED talk that should have transcripts
test_videos = [
    "mxYtd9DVIfU",  # Your original video
    "fKopy74weus",  # Popular TED talk
    "ZMxIy79d1eM",  # Another popular video
    "jNQXAC9IVRw"   # "Me at the zoo" - first YouTube video
]

for video_id in test_videos:
    print(f"\n🎬 Testing video: {video_id}")
    print("=" * 40)
    
    try:
        # Try the simplest method
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        print(f"✅ SUCCESS! Got {len(transcript_data)} entries")
        
        # Show first entry
        if transcript_data:
            first_entry = transcript_data[0]
            print(f"First line: [{first_entry['start']:.1f}s] {first_entry['text'][:100]}")
        
        # If this video works, we know the API is fine
        if video_id != "mxYtd9DVIfU":
            print(f"🎉 Video {video_id} works! The API is functional.")
            print("The issue is specific to your target video.")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        
        # If this is your original video, that's expected
        if video_id == "mxYtd9DVIfU":
            print("(This is your problem video - expected to fail)")

print("\n" + "=" * 50)
print("🔍 DIAGNOSIS:")
print("If ANY of the test videos worked, the problem is with your specific video.")
print("If ALL videos failed, there's an issue with your setup or YouTube is blocking the API.")
