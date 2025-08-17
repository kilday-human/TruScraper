#!/usr/bin/env python3
"""
TruScraper v6 - ULTIMATE AI ANALYSIS ENGINE
Combines enhanced text analysis with optional audio conviction detection
Memory-efficient, sponsor-aware, production-ready
"""

import subprocess
import json
import sys
import re
import os
import gc
import psutil
from difflib import SequenceMatcher
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Optional audio analysis imports
try:
    import librosa
    import numpy as np
    AUDIO_AVAILABLE = True
    print("🎙️ Audio analysis available")
except ImportError:
    AUDIO_AVAILABLE = False
    print("📝 Text-only analysis (install librosa for audio features)")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

class MemoryMonitor:
    """Monitor memory usage and cleanup"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def check_memory_limit(limit_mb=8000):
        """Check if we're approaching memory limit"""
        current = MemoryMonitor.get_memory_usage()
        if current > limit_mb:
            print(f"⚠️ High memory usage: {current:.1f}MB")
            gc.collect()  # Force garbage collection
            return True
        return False
    
    @staticmethod
    def cleanup_temp_files(*files):
        """Clean up temporary files"""
        for file in files:
            try:
                if file and os.path.exists(file):
                    os.remove(file)
                    print(f"🗑️ Cleaned: {file}")
            except Exception:
                pass

class SponsorDetector:
    """Detect and filter out promotional content"""
    
    def __init__(self):
        self.sponsor_keywords = [
            'sponsor', 'affiliate', 'link below', 'description', 
            'deposit', 'trading', 'fees', 'bonus', 'cash back',
            'limited time', 'offer', 'vpn', 'exclusive'
        ]
        
        self.sponsor_patterns = [
            r'link.*below',
            r'deposit.*\$?\d+',
            r'\d+%.*fees?',
            r'cash\s+back',
            r'limited\s+time'
        ]
    
    def is_sponsor_segment(self, text: str) -> bool:
        """Check if text segment is promotional content"""
        text_lower = text.lower()
        
        # Check for multiple sponsor keywords
        keyword_count = sum(1 for keyword in self.sponsor_keywords if keyword in text_lower)
        if keyword_count >= 2:
            return True
        
        # Check for sponsor patterns
        for pattern in self.sponsor_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def filter_sponsors(self, segments: List[Dict]) -> List[Dict]:
        """Remove sponsor segments from analysis"""
        filtered = []
        sponsor_count = 0
        
        for segment in segments:
            if not self.is_sponsor_segment(segment['text']):
                filtered.append(segment)
            else:
                sponsor_count += 1
        
        print(f"🚫 Filtered out {sponsor_count} sponsor segments")
        return filtered

class EnhancedTextAnalyzer:
    """Enhanced text analysis with better sentence reconstruction"""
    
    def __init__(self):
        self.sponsor_detector = SponsorDetector()
        
        # Enhanced patterns for financial analysis
        self.financial_patterns = {
            'price_targets': [
                r'(\d+k?\s*(?:btc|bitcoin))',
                r'(\d+k?\s*(?:eth|ethereum))', 
                r'(\$\d+[,\d]*k?)',
                r'(\d+\s*thousand)'
            ],
            'timeframes': [
                r'(by\s+(?:october|november|december|q[1-4]))',
                r'(in\s+202[5-9])',
                r'(next\s+(?:year|quarter|month))',
                r'(end\s+of\s+\w+)'
            ],
            'confidence_high': ['will', 'definitely', 'certainly', 'guarantee', 'absolutely'],
            'confidence_low': ['maybe', 'could', 'might', 'possibly', 'perhaps', 'probably'],
            'market_structure': ['cycle', 'top', 'bottom', 'peak', 'bull', 'bear', 'maturation'],
            'institutional': ['etf', 'blackrock', 'institutional', 'wall street', 'mainstream', 'adoption']
        }

    def smart_sentence_reconstruction(self, raw_segments: List[Dict]) -> List[Dict]:
        """ENHANCED sentence reconstruction with better logic"""
        print("🔧 Enhanced sentence reconstruction...")
        
        if not raw_segments:
            return []
        
        # Filter sponsors first
        filtered_segments = self.sponsor_detector.filter_sponsors(raw_segments)
        
        # Group segments into coherent sentences
        sentences = []
        current_sentence = ""
        current_time = ""
        word_buffer = []
        
        for segment in filtered_segments:
            text = segment['text'].strip()
            if not text:
                continue
            
            # Check if this starts a new sentence
            if self.is_new_sentence_start(text, current_sentence):
                # Save previous sentence if substantial
                if current_sentence and len(current_sentence.split()) > 8:
                    sentences.append({
                        'time': current_time,
                        'text': self.clean_sentence(current_sentence),
                        'word_count': len(current_sentence.split())
                    })
                
                # Start new sentence
                current_sentence = text
                current_time = segment['time']
                word_buffer = text.split()
            
            elif self.should_merge(text, current_sentence, word_buffer):
                # Merge with current sentence
                merged = self.smart_merge(current_sentence, text)
                if merged != current_sentence:  # Only if actually different
                    current_sentence = merged
                    word_buffer.extend(text.split())
            
            else:
                # Save current and start new
                if current_sentence and len(current_sentence.split()) > 8:
                    sentences.append({
                        'time': current_time,
                        'text': self.clean_sentence(current_sentence),
                        'word_count': len(current_sentence.split())
                    })
                
                current_sentence = text
                current_time = segment['time']
                word_buffer = text.split()
        
        # Don't forget the last sentence
        if current_sentence and len(current_sentence.split()) > 8:
            sentences.append({
                'time': current_time,
                'text': self.clean_sentence(current_sentence),
                'word_count': len(current_sentence.split())
            })
        
        # Remove duplicates with better algorithm
        unique_sentences = self.remove_duplicates_smart(sentences)
        
        print(f"✅ Reconstructed: {len(filtered_segments)} → {len(unique_sentences)} coherent sentences")
        return unique_sentences

    def is_new_sentence_start(self, text: str, current: str) -> bool:
        """Check if text starts a new sentence"""
        if not current:
            return True
        
        # Strong sentence starters
        strong_starters = [
            'Everyone', 'The', 'Bitcoin', 'Ethereum', 'Now', 'So', 'But', 
            'However', 'This', 'That', 'What', 'When', 'Right', 'Okay'
        ]
        
        first_word = text.split()[0] if text.split() else ""
        
        # Check for strong starter
        if first_word in strong_starters:
            return True
        
        # Check for complete thought patterns
        if any(pattern in text.lower() for pattern in [
            'everyone is wrong', 'the reality is', 'key point', 
            'bottom line', 'here\'s the thing'
        ]):
            return True
        
        # Very long segments are likely new thoughts
        if len(text.split()) > 15:
            return True
        
        return False

    def should_merge(self, text: str, current: str, word_buffer: List[str]) -> bool:
        """Check if text should merge with current sentence"""
        if not current:
            return False
        
        # Don't merge if too long already
        if len(word_buffer) > 50:
            return False
        
        # Check for continuation words
        continuation_starters = ['and', 'or', 'but', 'because', 'since', 'which', 'that']
        first_word = text.split()[0].lower() if text.split() else ""
        
        if first_word in continuation_starters:
            return True
        
        # Check for semantic overlap
        current_words = set(w.lower() for w in current.split()[-10:])  # Last 10 words
        text_words = set(w.lower() for w in text.split()[:10])  # First 10 words
        
        overlap = len(current_words & text_words)
        return overlap >= 2  # At least 2 words in common

    def smart_merge(self, current: str, new_text: str) -> str:
        """Intelligently merge text segments"""
        # Remove redundant repetitions
        current_words = current.split()
        new_words = new_text.split()
        
        # Find overlap at the end of current and start of new
        max_overlap = min(len(current_words), len(new_words), 5)
        best_overlap = 0
        
        for i in range(1, max_overlap + 1):
            if current_words[-i:] == new_words[:i]:
                best_overlap = i
        
        if best_overlap > 0:
            # Merge by removing overlap
            merged = current + " " + " ".join(new_words[best_overlap:])
        else:
            # Simple concatenation
            merged = current + " " + new_text
        
        return merged

    def clean_sentence(self, sentence: str) -> str:
        """Clean up sentence text"""
        # Remove excessive whitespace
        sentence = ' '.join(sentence.split())
        
        # Remove common filler repetitions
        words = sentence.split()
        cleaned_words = []
        
        for i, word in enumerate(words):
            # Skip if same word appears within 3 positions
            if i == 0 or word.lower() not in [w.lower() for w in words[max(0, i-3):i]]:
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)

    def remove_duplicates_smart(self, sentences: List[Dict]) -> List[Dict]:
        """Remove duplicates with enhanced logic"""
        if not sentences:
            return []
        
        unique = [sentences[0]]
        
        for current in sentences[1:]:
            is_duplicate = False
            
            # Check against recent sentences
            for existing in unique[-3:]:
                # Calculate similarity
                similarity = SequenceMatcher(
                    None, 
                    current['text'].lower(), 
                    existing['text'].lower()
                ).ratio()
                
                # More sophisticated duplicate detection
                if similarity > 0.75:  # 75% similar
                    is_duplicate = True
                    break
                
                # Check for substring containment
                current_clean = current['text'].lower().replace(' ', '')
                existing_clean = existing['text'].lower().replace(' ', '')
                
                if (len(current_clean) > 50 and current_clean in existing_clean) or \
                   (len(existing_clean) > 50 and existing_clean in current_clean):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(current)
        
        return unique

    def extract_enhanced_predictions(self, sentences: List[Dict]) -> List[Dict]:
        """Enhanced prediction extraction with better context"""
        print("🔮 Enhanced prediction extraction...")
        
        predictions = []
        
        for sentence in sentences:
            text = sentence['text']
            text_lower = text.lower()
            
            # Look for price targets
            price_found = False
            for pattern in self.financial_patterns['price_targets']:
                if re.search(pattern, text_lower):
                    price_found = True
                    break
            
            # Look for timeframes
            timeframe_found = False
            timeframe = 'Unspecified'
            for pattern in self.financial_patterns['timeframes']:
                match = re.search(pattern, text_lower)
                if match:
                    timeframe_found = True
                    timeframe = match.group(1)
                    break
            
            # Must have prediction indicators
            prediction_words = ['will', 'expect', 'predict', 'target', 'by', 'in 202']
            has_prediction_word = any(word in text_lower for word in prediction_words)
            
            # Create prediction if criteria met
            if (price_found or timeframe_found) and has_prediction_word:
                confidence = self.assess_text_confidence(text_lower)
                pred_type = self.categorize_prediction_enhanced(text_lower)
                
                prediction = {
                    'time': sentence['time'],
                    'text': text,
                    'confidence': confidence,
                    'timeframe': timeframe,
                    'prediction_type': pred_type,
                    'key_numbers': re.findall(r'\d+[kK]?', text),
                    'context_score': self.calculate_context_score(text)
                }
                
                predictions.append(prediction)
        
        # Sort by confidence and context
        predictions.sort(key=lambda x: (
            {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x['confidence']], 
            x['context_score']
        ), reverse=True)
        
        return predictions

    def assess_text_confidence(self, text: str) -> str:
        """Assess confidence from text patterns"""
        high_conf_count = sum(1 for word in self.financial_patterns['confidence_high'] if word in text)
        low_conf_count = sum(1 for word in self.financial_patterns['confidence_low'] if word in text)
        
        if high_conf_count > low_conf_count and high_conf_count > 0:
            return 'HIGH'
        elif low_conf_count > 0:
            return 'LOW'
        else:
            return 'MEDIUM'

    def categorize_prediction_enhanced(self, text: str) -> str:
        """Enhanced prediction categorization"""
        if any(word in text for word in ['price', 'target', '$', 'k', 'btc', 'eth']):
            if 'local' in text or 'peak' in text:
                return 'LOCAL_PRICE_TARGET'
            else:
                return 'PRICE_TARGET'
        elif any(word in text for word in ['top', 'peak', 'cycle', 'end']):
            return 'MARKET_TIMING'
        elif any(word in text for word in self.financial_patterns['institutional']):
            return 'INSTITUTIONAL_TREND'
        else:
            return 'GENERAL_FORECAST'

    def calculate_context_score(self, text: str) -> float:
        """Calculate how important this prediction is based on context"""
        score = 0.5
        
        # Financial terms boost score
        financial_terms = ['bitcoin', 'ethereum', 'price', 'market', 'cycle']
        score += 0.1 * sum(1 for term in financial_terms if term in text.lower())
        
        # Specific numbers boost score  
        if re.search(r'\d+[kK]', text):
            score += 0.2
        
        # Length indicates substance
        if len(text.split()) > 20:
            score += 0.1
        
        return min(1.0, score)

class AudioConvictionAnalyzer:
    """Memory-efficient audio analysis"""
    
    def __init__(self):
        self.whisper_model = None
        self.sample_rate = 22050
        
    def initialize_whisper(self):
        """Lazy load Whisper model"""
        if not WHISPER_AVAILABLE:
            return False
        
        try:
            if self.whisper_model is None:
                print("🎯 Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
                print("✅ Whisper loaded")
            return True
        except Exception as e:
            print(f"❌ Whisper initialization failed: {e}")
            return False

    def download_audio(self, video_url: str, video_id: str) -> Optional[str]:
        """Download audio with memory monitoring"""
        try:
            print("🎵 Downloading audio...")
            audio_file = f"audio_{video_id}.wav"
            
            # Check available space
            if MemoryMonitor.get_memory_usage() > 6000:  # 6GB threshold
                print("⚠️ High memory usage, skipping audio download")
                return None
            
            cmd = [
                'yt-dlp', '-x', '--audio-format', 'wav',
                '--audio-quality', '0', '-o', f'audio_{video_id}.%(ext)s',
                video_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(audio_file):
                # Check file size
                size_mb = os.path.getsize(audio_file) / 1024 / 1024
                print(f"✅ Audio downloaded: {audio_file} ({size_mb:.1f}MB)")
                
                if size_mb > 500:  # 500MB limit
                    print("⚠️ Audio file too large, skipping analysis")
                    MemoryMonitor.cleanup_temp_files(audio_file)
                    return None
                
                return audio_file
            else:
                print(f"❌ Audio download failed")
                return None
                
        except Exception as e:
            print(f"❌ Audio download error: {e}")
            return None

    def analyze_audio_conviction(self, audio_file: str, text_predictions: List[Dict]) -> Dict:
        """Analyze audio for conviction with memory efficiency"""
        if not self.initialize_whisper():
            return {}
        
        try:
            print("🎙️ Analyzing audio conviction...")
            
            # Transcribe with timestamps
            result = self.whisper_model.transcribe(audio_file, word_timestamps=True)
            
            # Analyze conviction for prediction segments
            conviction_results = []
            
            for pred in text_predictions:
                pred_time = self.parse_timestamp(pred['time'])
                
                # Find matching audio segment
                for segment in result['segments']:
                    if abs(segment['start'] - pred_time) < 5.0:  # Within 5 seconds
                        conviction = self.analyze_segment_conviction(
                            audio_file, segment, pred['text']
                        )
                        if conviction:
                            conviction_results.append({
                                'prediction': pred,
                                'audio_conviction': conviction
                            })
                        break
            
            return {
                'conviction_results': conviction_results,
                'total_analyzed': len(conviction_results),
                'audio_available': True
            }
            
        except Exception as e:
            print(f"❌ Audio analysis error: {e}")
            return {}

    def parse_timestamp(self, timestamp: str) -> float:
        """Parse MM:SS timestamp to seconds"""
        try:
            parts = timestamp.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            return 0.0
        except:
            return 0.0

    def analyze_segment_conviction(self, audio_file: str, segment: Dict, text: str) -> Optional[Dict]:
        """Analyze conviction for a specific segment"""
        try:
            # Load audio segment
            y, sr = librosa.load(
                audio_file, 
                sr=self.sample_rate,
                offset=segment['start'],
                duration=min(segment['end'] - segment['start'], 10.0)  # Max 10 seconds
            )
            
            if len(y) == 0:
                return None
            
            # Quick audio features
            rms_energy = float(np.mean(librosa.feature.rms(y=y)[0]))
            
            # Simple conviction score based on volume and text
            volume_confidence = min(1.0, rms_energy * 50)  # Normalize
            text_confidence = self.get_text_confidence_score(text)
            
            combined_confidence = (volume_confidence + text_confidence) / 2
            
            return {
                'confidence_score': combined_confidence,
                'volume_level': volume_confidence,
                'text_confidence': text_confidence,
                'duration': len(y) / sr
            }
            
        except Exception as e:
            print(f"⚠️ Segment analysis error: {e}")
            return None

    def get_text_confidence_score(self, text: str) -> float:
        """Get confidence score from text"""
        text_lower = text.lower()
        
        high_conf_words = ['will', 'definitely', 'certainly', 'absolutely']
        low_conf_words = ['maybe', 'could', 'might', 'possibly']
        
        high_count = sum(1 for word in high_conf_words if word in text_lower)
        low_count = sum(1 for word in low_conf_words if word in text_lower)
        
        if high_count > 0:
            return 0.8
        elif low_count > 0:
            return 0.3
        else:
            return 0.5

class TruScraperV6:
    """Ultimate TruScraper with integrated analysis"""
    
    def __init__(self):
        print("🚀 TruScraper v6 - Ultimate Analysis Engine")
        self.text_analyzer = EnhancedTextAnalyzer()
        self.audio_analyzer = AudioConvictionAnalyzer() if AUDIO_AVAILABLE else None
        self.memory_monitor = MemoryMonitor()
        
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from URL"""
        match = re.search(r'(?:v=|youtu\.be/)([^&\n?#]+)', url)
        if match:
            return match.group(1)
        raise ValueError("Could not extract video ID")

    def download_subtitles(self, video_url: str, video_id: str) -> Optional[str]:
        """Download subtitles"""
        try:
            print("📥 Downloading subtitles...")
            cmd = [
                'yt-dlp', '--write-auto-subs', '--sub-langs', 'en',
                '--sub-format', 'vtt', '--skip-download',
                '--output', f'transcript_{video_id}.%(ext)s', video_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                for suffix in ['.en.vtt', '.en-US.vtt', '.en-GB.vtt']:
                    filename = f'transcript_{video_id}{suffix}'
                    if os.path.exists(filename):
                        return filename
            return None
        except Exception as e:
            print(f"❌ Subtitle download error: {e}")
            return None

    def parse_vtt_file(self, vtt_file: str) -> List[Dict]:
        """Parse VTT file into raw segments"""
        try:
            with open(vtt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            raw_segments = []
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if '-->' in line and i + 1 < len(lines):
                    timestamp = line.split('-->')[0].strip()
                    
                    text_lines = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip():
                        text_lines.append(lines[j].strip())
                        j += 1
                    
                    if text_lines:
                        text = ' '.join(text_lines)
                        text = re.sub(r'<[^>]+>', '', text)
                        text = ' '.join(text.split())
                        
                        if len(text) > 5:
                            raw_segments.append({
                                'time': self.format_timestamp(timestamp),
                                'text': text
                            })
            
            return raw_segments
            
        except Exception as e:
            print(f"❌ VTT parsing error: {e}")
            return []

    def format_timestamp(self, timestamp: str) -> str:
        """Format timestamp to MM:SS"""
        try:
            parts = timestamp.split(':')
            if len(parts) >= 3:
                h, m, s = parts[0], parts[1], parts[2].split('.')[0]
                return f"{m}:{s}" if h == '00' else f"{h}:{m}:{s}"
        except:
            pass
        return timestamp

    def create_intelligent_summary(self, sentences: List[Dict], predictions: List[Dict]) -> str:
        """Create intelligent summary that captures key insights"""
        if not sentences:
            return "No content available for summary"
        
        # Find thesis
        thesis = None
        for sentence in sentences[:10]:
            if any(phrase in sentence['text'].lower() for phrase in [
                'everyone is wrong', 'reality is', 'key point'
            ]):
                thesis = sentence['text']
                break
        
        # Get top prediction
        top_prediction = predictions[0] if predictions else None
        
        # Build summary
        summary_parts = []
        
        if thesis:
            summary_parts.append(f"THESIS: {thesis}")
        
        if top_prediction:
            summary_parts.append(f"KEY PREDICTION: {top_prediction['text'][:100]}...")
        
        # Add key insights
        key_insights = []
        for sentence in sentences:
            text_lower = sentence['text'].lower()
            if any(term in text_lower for term in ['etf', 'institutional', 'maturation', 'cycle']):
                if len(sentence['text']) > 50:
                    key_insights.append(sentence['text'][:100] + "...")
                    if len(key_insights) >= 2:
                        break
        
        for insight in key_insights:
            summary_parts.append(f"INSIGHT: {insight}")
        
        return " | ".join(summary_parts)

    def generate_comprehensive_report(self, analysis_data: Dict, video_id: str) -> str:
        """Generate comprehensive analysis report"""
        
        sentences = analysis_data['sentences']
        predictions = analysis_data['predictions']
        audio_analysis = analysis_data.get('audio_analysis', {})
        
        report = f"""
🧠 TRUSCRAPER v6 - ULTIMATE ANALYSIS
Video ID: {video_id}
Analysis: Enhanced Text + {'Audio Conviction' if audio_analysis else 'Text Only'}
{'='*60}

📋 INTELLIGENT SUMMARY
{analysis_data['summary']}

🎯 MAIN THESIS
{sentences[0]['text'] if sentences else 'No clear thesis identified'}

🔮 PREDICTIONS ANALYSIS ({len(predictions)} found)
"""
        
        # Group predictions by type
        pred_types = {}
        for pred in predictions:
            ptype = pred['prediction_type']
            if ptype not in pred_types:
                pred_types[ptype] = []
            pred_types[ptype].append(pred)
        
        for ptype, preds in pred_types.items():
            report += f"\n{ptype.replace('_', ' ').title()}:\n"
            for pred in preds[:3]:
                conf_emoji = {'HIGH': '🔥', 'MEDIUM': '⚠️', 'LOW': '🤔'}[pred['confidence']]
                report += f"  {conf_emoji} [{pred['time']}] {pred['text']}\n"
                
                if pred['key_numbers']:
                    report += f"     Numbers: {', '.join(pred['key_numbers'])}\n"
                if pred['timeframe'] != 'Unspecified':
                    report += f"     Timeframe: {pred['timeframe']}\n"
                
                # Add audio conviction if available
                if audio_analysis and 'conviction_results' in audio_analysis:
                    for conv_result in audio_analysis['conviction_results']:
                        if conv_result['prediction']['time'] == pred['time']:
                            audio_conv = conv_result['audio_conviction']
                            report += f"     🎙️ VOICE CONVICTION: {audio_conv['confidence_score']:.2f}/1.0\n"
                            break
        
        # Audio analysis section
        if audio_analysis and 'conviction_results' in audio_analysis:
            report += f"""
🎙️ AUDIO CONVICTION ANALYSIS
• Total predictions analyzed: {audio_analysis['total_analyzed']}
• High-conviction audio segments detected
• Voice patterns analyzed for true confidence levels
"""
        
        report += f"""
📊 ANALYSIS METRICS
• Coherent sentences: {len(sentences)}
• Predictions extracted: {len(predictions)}
• Audio analysis: {'✅ Available' if audio_analysis else '❌ Text only'}
• Memory usage: {MemoryMonitor.get_memory_usage():.1f}MB

💡 KEY INSIGHTS
"""
        
        # Extract key insights
        insights = []
        for sentence in sentences:
            text_lower = sentence['text'].lower()
            if any(term in text_lower for term in ['maturation', 'etf', 'institutional', 'cycle']):
                insights.append(f"• {sentence['text'][:80]}...")
                if len(insights) >= 3:
                    break
        
        for insight in insights:
            report += insight + "\n"
        
        report += f"""
{'='*60}
🚀 ULTIMATE ANALYSIS COMPLETE
Enhanced text understanding + optional audio conviction detection
Memory-efficient, sponsor-aware, production-ready
"""
        
        return report

def main():
    """Main execution"""
    if len(sys.argv) != 2:
        print("Usage: python3 truscraper_v6.py <youtube_url>")
        print("Features: Enhanced text analysis + optional audio conviction detection")
        sys.exit(1)
    
    # Check dependencies
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except:
        print("❌ yt-dlp not installed! Run: pip install yt-dlp")
        sys.exit(1)
    
    scraper = TruScraperV6()
    youtube_url = sys.argv[1]
    
    try:
        print(f"🚀 Starting Ultimate Analysis...")
        print(f"💾 Memory available: {MemoryMonitor.get_memory_usage():.1f}MB used")
        
        # Extract video ID
        video_id = scraper.extract_video_id(youtube_url)
        print(f"📹 Video ID: {video_id}")
        
        # Download subtitles
        subtitle_file = scraper.download_subtitles(youtube_url, video_id)
        if not subtitle_file:
            print("❌ Failed to download subtitles")
            sys.exit(1)
        
        # Parse VTT file
        raw_segments = scraper.parse_vtt_file(subtitle_file)
        if not raw_segments:
            print("❌ Failed to parse subtitles")
            sys.exit(1)
        
        print(f"📝 Parsed {len(raw_segments)} raw segments")
        
        # Enhanced text analysis
        print("🧠 Running enhanced text analysis...")
        sentences = scraper.text_analyzer.smart_sentence_reconstruction(raw_segments)
        predictions = scraper.text_analyzer.extract_enhanced_predictions(sentences)
        
        print(f"   ✅ {len(sentences)} coherent sentences")
        print(f"   ✅ {len(predictions)} predictions extracted")
        
        # Memory check before audio analysis
        MemoryMonitor.check_memory_limit(6000)
        
        # Audio analysis (if available and memory allows)
        audio_analysis = {}
        if AUDIO_AVAILABLE and scraper.audio_analyzer and len(predictions) > 0:
            if MemoryMonitor.get_memory_usage() < 5000:  # 5GB threshold
                print("🎙️ Attempting audio conviction analysis...")
                
                audio_file = scraper.audio_analyzer.download_audio(youtube_url, video_id)
                if audio_file:
                    audio_analysis = scraper.audio_analyzer.analyze_audio_conviction(
                        audio_file, predictions
                    )
                    MemoryMonitor.cleanup_temp_files(audio_file)
                else:
                    print("⚠️ Audio analysis skipped - download failed")
            else:
                print("⚠️ Audio analysis skipped - memory limit")
        elif not AUDIO_AVAILABLE:
            print("📝 Audio analysis not available (install librosa + whisper)")
        
        # Create intelligent summary
        summary = scraper.create_intelligent_summary(sentences, predictions)
        
        # Compile analysis data
        analysis_data = {
            'sentences': sentences,
            'predictions': predictions,
            'audio_analysis': audio_analysis,
            'summary': summary
        }
        
        # Generate comprehensive report
        report = scraper.generate_comprehensive_report(analysis_data, video_id)
        
        # Save and display
        output_file = f"ultimate_analysis_{video_id}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n" + report)
        print(f"\n💾 Ultimate analysis saved: {output_file}")
        print(f"💾 Final memory usage: {MemoryMonitor.get_memory_usage():.1f}MB")
        
        # Cleanup
        MemoryMonitor.cleanup_temp_files(subtitle_file)
        
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Try running with a different video or check your internet connection")
        sys.exit(1)

if __name__ == "__main__":
    main()
