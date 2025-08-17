#!/usr/bin/env python3
"""
TruScraper v4 - Intelligent transcript analysis (CLEAN & ROBUST VERSION)
Simplified, debuggable, and optimized for reliability
"""

import subprocess
import json
import sys
import re
import os
from difflib import SequenceMatcher

class TranscriptAnalyzer:
    def __init__(self):
        # Core keywords for analysis
        self.prediction_words = ['will', 'expect', 'predict', 'target', 'by', 'in 202']
        self.confidence_high = ['definitely', 'certainly', 'absolutely', 'guarantee']
        self.confidence_low = ['maybe', 'could', 'might', 'possibly']
        self.contrarian_words = ['wrong', 'everyone', 'crowd', 'mistake', 'opposite']
        
        print("🧠 AI Analyzer initialized")

    def extract_video_id(self, url):
        """Extract video ID - robust version"""
        try:
            match = re.search(r'(?:v=|youtu\.be/)([^&\n?#]+)', url)
            if match:
                return match.group(1)
            raise ValueError("Could not find video ID")
        except Exception as e:
            print(f"❌ URL parsing error: {e}")
            raise

    def download_subtitles(self, video_url, video_id):
        """Download subtitles with proper error handling"""
        try:
            print("📥 Downloading subtitles...")
            cmd = [
                'yt-dlp', '--write-auto-subs', '--sub-langs', 'en', 
                '--sub-format', 'vtt', '--skip-download',
                '--output', f'transcript_{video_id}.%(ext)s', video_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"❌ yt-dlp failed: {result.stderr}")
                return None
            
            # Find the downloaded file
            for suffix in ['.en.vtt', '.en-US.vtt', '.en-GB.vtt']:
                filename = f'transcript_{video_id}{suffix}'
                if os.path.exists(filename):
                    print(f"✅ Downloaded: {filename}")
                    return filename
            
            print("❌ No subtitle file found after download")
            return None
            
        except subprocess.TimeoutExpired:
            print("❌ Download timeout - video may be too long or restricted")
            return None
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None

    def parse_vtt_clean(self, vtt_file):
        """Parse VTT with smart deduplication - simplified version"""
        try:
            print("🧹 Parsing and cleaning transcript...")
            
            with open(vtt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract segments with timestamps
            segments = []
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if '-->' in line and i + 1 < len(lines):
                    # Get timestamp
                    timestamp = line.split('-->')[0].strip()
                    
                    # Get text from next non-empty lines
                    text_lines = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip():
                        text_lines.append(lines[j].strip())
                        j += 1
                    
                    if text_lines:
                        # Clean the text
                        text = ' '.join(text_lines)
                        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
                        text = ' '.join(text.split())  # Clean whitespace
                        
                        # Only keep meaningful segments
                        if len(text) > 10 and len(text.split()) >= 3:
                            segments.append({
                                'time': self.format_timestamp(timestamp),
                                'text': text
                            })
            
            # Smart deduplication
            return self.remove_duplicates(segments)
            
        except Exception as e:
            print(f"❌ Parsing error: {e}")
            return None

    def format_timestamp(self, timestamp):
        """Convert timestamp to MM:SS format"""
        try:
            # Extract time part and convert
            time_part = timestamp.split()[0] if ' ' in timestamp else timestamp
            parts = time_part.split(':')
            if len(parts) >= 3:
                h, m, s = parts[0], parts[1], parts[2].split('.')[0]
                return f"{m}:{s}" if h == '00' else f"{h}:{m}:{s}"
            return timestamp
        except:
            return timestamp

    def remove_duplicates(self, segments):
        """Remove duplicate segments efficiently"""
        if not segments:
            return []
        
        cleaned = [segments[0]]
        
        for current in segments[1:]:
            # Check similarity with last few segments
            is_duplicate = False
            for prev in cleaned[-3:]:  # Only check last 3
                similarity = SequenceMatcher(None, 
                    current['text'].lower(), 
                    prev['text'].lower()
                ).ratio()
                
                if similarity > 0.8:  # 80% similar
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                cleaned.append(current)
        
        print(f"✅ Cleaned: {len(segments)} → {len(cleaned)} segments")
        return cleaned

    def extract_predictions(self, segments):
        """Extract predictions and price targets"""
        predictions = []
        
        for segment in segments:
            text = segment['text']
            text_lower = text.lower()
            
            # Look for price predictions
            price_patterns = [
                r'(\d+k?\s*(btc|bitcoin|eth|ethereum))',
                r'(\$\d+[,\d]*)',
                r'(target\s+\d+)',
                r'(\d+\s*thousand)',
                r'(expect.*\d+)'
            ]
            
            has_prediction = any(re.search(pattern, text_lower) for pattern in price_patterns)
            has_prediction_word = any(word in text_lower for word in self.prediction_words)
            
            if has_prediction and has_prediction_word:
                confidence = self.get_confidence_level(text_lower)
                predictions.append({
                    'time': segment['time'],
                    'text': text,
                    'confidence': confidence
                })
        
        return predictions

    def extract_key_claims(self, segments):
        """Extract important claims and insights"""
        claims = []
        
        for segment in segments:
            text = segment['text']
            text_lower = text.lower()
            
            # Look for strong statements
            is_claim = (
                any(word in text_lower for word in self.contrarian_words) or
                'key point' in text_lower or
                'important' in text_lower or
                'reality is' in text_lower or
                'truth is' in text_lower
            )
            
            if is_claim:
                strength = self.get_claim_strength(text_lower)
                claims.append({
                    'time': segment['time'],
                    'text': text,
                    'strength': strength
                })
        
        return claims

    def get_confidence_level(self, text):
        """Assess confidence level in statement"""
        if any(word in text for word in self.confidence_high):
            return 'HIGH'
        elif any(word in text for word in self.confidence_low):
            return 'LOW'
        else:
            return 'MEDIUM'

    def get_claim_strength(self, text):
        """Assess strength of claim"""
        strong_indicators = ['absolutely', 'definitely', 'guarantee', 'certain']
        weak_indicators = ['maybe', 'possibly', 'might', 'could']
        
        if any(word in text for word in strong_indicators):
            return 'STRONG'
        elif any(word in text for word in weak_indicators):
            return 'WEAK'
        else:
            return 'MODERATE'

    def find_main_thesis(self, segments):
        """Find the main thesis/argument"""
        thesis_indicators = ['everyone is wrong', 'main point', 'key insight', 'reality is']
        
        for segment in segments[:10]:  # Check first 10 segments
            text_lower = segment['text'].lower()
            if any(indicator in text_lower for indicator in thesis_indicators):
                return {
                    'time': segment['time'],
                    'text': segment['text']
                }
        
        # Fallback: return first substantial segment
        for segment in segments:
            if len(segment['text']) > 50:
                return {
                    'time': segment['time'],
                    'text': segment['text']
                }
        
        return {'time': '00:00', 'text': 'No clear thesis identified'}

    def create_summary(self, segments):
        """Create intelligent summary"""
        if len(segments) < 5:
            return "Video too short for meaningful summary."
        
        # Get key sentences from different parts
        total = len(segments)
        key_segments = [
            segments[0],  # Opening
            segments[total//4],  # Early
            segments[total//2],  # Middle  
            segments[3*total//4],  # Late
            segments[-1]  # Ending
        ]
        
        summary_parts = []
        for seg in key_segments:
            if len(seg['text']) > 30:  # Only substantial content
                summary_parts.append(seg['text'])
        
        return ' '.join(summary_parts[:3])  # Keep it concise

    def generate_analysis(self, segments, predictions, claims, thesis):
        """Generate final analysis report"""
        # Count key themes
        all_text = ' '.join([seg['text'] for seg in segments]).lower()
        
        themes = {}
        theme_words = {
            'bitcoin': ['bitcoin', 'btc'],
            'ethereum': ['ethereum', 'eth'],
            'market_cycle': ['cycle', 'bull', 'bear', 'top'],
            'institutional': ['institutional', 'etf', 'wall street'],
            'predictions': ['will', 'expect', 'target', 'predict']
        }
        
        for theme, words in theme_words.items():
            count = sum(all_text.count(word) for word in words)
            if count > 0:
                themes[theme] = count
        
        # Risk assessment
        risks = []
        high_conf_predictions = [p for p in predictions if p['confidence'] == 'HIGH']
        if len(high_conf_predictions) > 2:
            risks.append("⚠️ Multiple high-confidence predictions detected")
        
        return {
            'thesis': thesis,
            'summary': self.create_summary(segments),
            'predictions': predictions[:5],  # Top 5
            'claims': claims[:5],  # Top 5
            'themes': themes,
            'risks': risks,
            'total_segments': len(segments)
        }

    def format_report(self, analysis, video_id):
        """Format analysis into readable report"""
        report = f"""
🧠 INTELLIGENT ANALYSIS REPORT
Video ID: {video_id}
{'='*50}

📋 MAIN THESIS
[{analysis['thesis']['time']}] {analysis['thesis']['text']}

📝 EXECUTIVE SUMMARY
{analysis['summary']}

🔮 KEY PREDICTIONS ({len(analysis['predictions'])} found)
"""
        
        for pred in analysis['predictions']:
            conf_emoji = {'HIGH': '🔥', 'MEDIUM': '⚠️', 'LOW': '🤔'}[pred['confidence']]
            report += f"{conf_emoji} [{pred['time']}] {pred['text']}\n"
        
        report += f"\n💡 IMPORTANT CLAIMS ({len(analysis['claims'])} found)\n"
        
        for claim in analysis['claims']:
            strength_emoji = {'STRONG': '💪', 'MODERATE': '👍', 'WEAK': '🤷'}[claim['strength']]
            report += f"{strength_emoji} [{claim['time']}] {claim['text']}\n"
        
        report += f"\n📊 KEY THEMES\n"
        for theme, count in sorted(analysis['themes'].items(), key=lambda x: x[1], reverse=True):
            report += f"• {theme.upper()}: {count} mentions\n"
        
        if analysis['risks']:
            report += f"\n⚠️ RISK ASSESSMENT\n"
            for risk in analysis['risks']:
                report += f"• {risk}\n"
        
        report += f"\n📈 ANALYSIS STATS\n"
        report += f"• Total segments analyzed: {analysis['total_segments']}\n"
        report += f"• Predictions extracted: {len(analysis['predictions'])}\n"
        report += f"• Claims identified: {len(analysis['claims'])}\n"
        
        report += "\n" + "="*50
        report += "\n✅ Analysis complete!"
        
        return report

def main():
    """Main execution with proper error handling"""
    if len(sys.argv) != 2:
        print("Usage: python3 scraper_v4.py <youtube_url>")
        sys.exit(1)
    
    # Check dependencies
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ yt-dlp not installed! Run: pip install yt-dlp")
        sys.exit(1)
    
    analyzer = TranscriptAnalyzer()
    youtube_url = sys.argv[1]
    
    try:
        print("🚀 Starting TruScraper v4 AI Analysis...")
        
        # Step 1: Extract video ID
        video_id = analyzer.extract_video_id(youtube_url)
        print(f"📹 Video ID: {video_id}")
        
        # Step 2: Download subtitles
        subtitle_file = analyzer.download_subtitles(youtube_url, video_id)
        if not subtitle_file:
            print("❌ Failed to download subtitles")
            sys.exit(1)
        
        # Step 3: Parse and clean
        segments = analyzer.parse_vtt_clean(subtitle_file)
        if not segments:
            print("❌ Failed to parse transcript")
            sys.exit(1)
        
        # Step 4: AI Analysis
        print("🤖 Running AI analysis...")
        predictions = analyzer.extract_predictions(segments)
        claims = analyzer.extract_key_claims(segments)
        thesis = analyzer.find_main_thesis(segments)
        
        print(f"   Found {len(predictions)} predictions")
        print(f"   Found {len(claims)} key claims")
        
        # Step 5: Generate report
        analysis = analyzer.generate_analysis(segments, predictions, claims, thesis)
        report = analyzer.format_report(analysis, video_id)
        
        # Step 6: Save and display
        output_file = f"analysis_{video_id}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n" + report)
        print(f"\n💾 Report saved: {output_file}")
        
        # Cleanup
        try:
            os.remove(subtitle_file)
            print(f"🗑️ Cleaned up: {subtitle_file}")
        except:
            pass
        
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Try running with a different video or check your internet connection")
        sys.exit(1)

if __name__ == "__main__":
    main()
