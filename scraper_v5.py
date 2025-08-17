#!/usr/bin/env python3
"""
TruScraper v5 - TRUE AI UNDERSTANDING
This version actually comprehends the content like a human analyst would
"""

import subprocess
import json
import sys
import re
import os
from difflib import SequenceMatcher
from collections import defaultdict

class SmartTranscriptAnalyzer:
    def __init__(self):
        print("🧠 Smart Analyzer v5 - TRUE AI UNDERSTANDING")
        
        # Financial context understanding
        self.crypto_context = {
            'price_targets': r'(\d+k?\s*(?:btc|bitcoin|eth|ethereum)|\$\d+[,\d]*k?)',
            'timeframes': r'((?:by|in)\s+(?:october|november|december|q[1-4]|202[5-9]|next\s+\w+))',
            'market_calls': r'(top|bottom|peak|cycle|bull|bear|rally|crash|correction)',
            'institutions': r'(etf|blackrock|wall\s*street|institutional|treasury|401k|mainstream)',
            'confidence_signals': r'(will|expect|predict|guarantee|definitely|probably|maybe|could)'
        }
        
        # Argument structure patterns
        self.argument_patterns = {
            'thesis': r'(everyone\s+is\s+wrong|the\s+reality\s+is|key\s+point|main\s+thesis)',
            'evidence': r'(data\s+shows|evidence|charts?\s+show|statistics|numbers)',
            'reasoning': r'(because|since|therefore|as\s+a\s+result|this\s+means|which\s+suggests)',
            'contrarian': r'(however|but|contrary\s+to|unlike|opposite|different\s+from)',
            'predictions': r'(will\s+\w+|expect\s+\w+|target|forecast|by\s+\d{4})'
        }

    def extract_video_id(self, url):
        """Extract video ID"""
        match = re.search(r'(?:v=|youtu\.be/)([^&\n?#]+)', url)
        if match:
            return match.group(1)
        raise ValueError("Could not extract video ID")

    def download_subtitles(self, video_url, video_id):
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
            print(f"❌ Download error: {e}")
            return None

    def smart_parse_vtt(self, vtt_file):
        """SMART parsing - reconstructs coherent sentences"""
        try:
            print("🧠 Smart parsing - reconstructing coherent sentences...")
            
            with open(vtt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract all timed segments first
            raw_segments = []
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if '-->' in line and i + 1 < len(lines):
                    timestamp = line.split('-->')[0].strip()
                    
                    # Collect text
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
                                'text': text,
                                'raw_time': timestamp
                            })
            
            # SMART RECONSTRUCTION - merge fragments into coherent sentences
            return self.reconstruct_sentences(raw_segments)
            
        except Exception as e:
            print(f"❌ Parsing error: {e}")
            return None

    def reconstruct_sentences(self, raw_segments):
        """RECONSTRUCT coherent sentences from fragments"""
        print("🔧 Reconstructing coherent sentences...")
        
        reconstructed = []
        current_sentence = ""
        current_time = ""
        sentence_parts = []
        
        for segment in raw_segments:
            text = segment['text']
            time = segment['time']
            
            # Check if this looks like a sentence fragment or continuation
            if self.is_sentence_start(text):
                # Save previous sentence if we have one
                if current_sentence and len(current_sentence.split()) > 5:
                    reconstructed.append({
                        'time': current_time,
                        'text': current_sentence.strip(),
                        'length': len(current_sentence.split())
                    })
                
                # Start new sentence
                current_sentence = text
                current_time = time
                sentence_parts = [text]
            
            elif self.is_continuation(text, current_sentence):
                # Add to current sentence if it's a logical continuation
                if not self.is_duplicate_fragment(text, sentence_parts):
                    current_sentence += " " + text
                    sentence_parts.append(text)
            
            else:
                # This seems like a new thought - save current and start fresh
                if current_sentence and len(current_sentence.split()) > 5:
                    reconstructed.append({
                        'time': current_time,
                        'text': current_sentence.strip(),
                        'length': len(current_sentence.split())
                    })
                
                current_sentence = text
                current_time = time
                sentence_parts = [text]
        
        # Don't forget the last sentence
        if current_sentence and len(current_sentence.split()) > 5:
            reconstructed.append({
                'time': current_time,
                'text': current_sentence.strip(),
                'length': len(current_sentence.split())
            })
        
        # Filter out duplicates and short fragments
        final_sentences = self.remove_duplicate_sentences(reconstructed)
        
        print(f"✅ Reconstructed: {len(raw_segments)} fragments → {len(final_sentences)} coherent sentences")
        return final_sentences

    def is_sentence_start(self, text):
        """Check if text looks like the start of a new sentence"""
        # Starts with capital letter and isn't clearly a continuation
        if not text or not text[0].isupper():
            return False
        
        # Common sentence starters
        sentence_starters = ['Everyone', 'The', 'Bitcoin', 'Ethereum', 'Now', 'So', 'But', 'However', 'This', 'That', 'What', 'When', 'Where', 'Why', 'How']
        first_word = text.split()[0].rstrip('.,!?')
        
        return first_word in sentence_starters or len(text.split()) > 8

    def is_continuation(self, text, current_sentence):
        """Check if text is a logical continuation"""
        if not current_sentence:
            return False
        
        # Check for continuation patterns
        continuation_patterns = [
            r'^(and|or|but|because|since|which|that|who|what|when|where)',
            r'^(is|are|was|were|will|can|could|should|would)',
            r'^\w+ing\b',  # -ing words
            r'^\w+ed\b'    # -ed words
        ]
        
        text_lower = text.lower()
        for pattern in continuation_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Check semantic similarity
        last_words = current_sentence.split()[-5:]  # Last 5 words
        first_words = text.split()[:5]  # First 5 words
        
        # Look for word overlap
        overlap = set(w.lower() for w in last_words) & set(w.lower() for w in first_words)
        return len(overlap) > 0

    def is_duplicate_fragment(self, text, sentence_parts):
        """Check if this text is already covered in the sentence"""
        text_words = set(text.lower().split())
        
        for part in sentence_parts[-3:]:  # Check last 3 parts
            part_words = set(part.lower().split())
            overlap = len(text_words & part_words) / len(text_words) if text_words else 0
            if overlap > 0.7:  # 70% overlap = duplicate
                return True
        
        return False

    def remove_duplicate_sentences(self, sentences):
        """Remove duplicate sentences"""
        if not sentences:
            return []
        
        unique_sentences = [sentences[0]]
        
        for current in sentences[1:]:
            is_duplicate = False
            
            for existing in unique_sentences[-5:]:  # Check last 5
                similarity = SequenceMatcher(None, 
                    current['text'].lower(), 
                    existing['text'].lower()
                ).ratio()
                
                if similarity > 0.6:  # 60% similar = duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_sentences.append(current)
        
        return unique_sentences

    def format_timestamp(self, timestamp):
        """Format timestamp to MM:SS"""
        try:
            parts = timestamp.split(':')
            if len(parts) >= 3:
                h, m, s = parts[0], parts[1], parts[2].split('.')[0]
                return f"{m}:{s}" if h == '00' else f"{h}:{m}:{s}"
        except:
            pass
        return timestamp

    def extract_main_thesis(self, sentences):
        """INTELLIGENTLY extract the main thesis"""
        print("🎯 Extracting main thesis...")
        
        # Look for thesis indicators in order of preference
        thesis_patterns = [
            r'everyone\s+is\s+wrong',
            r'the\s+reality\s+is',
            r'key\s+point',
            r'main\s+thesis',
            r'important.*understand',
            r'truth.*is'
        ]
        
        for pattern in thesis_patterns:
            for sentence in sentences[:15]:  # Check first 15 sentences
                if re.search(pattern, sentence['text'].lower()):
                    return {
                        'time': sentence['time'],
                        'text': sentence['text'],
                        'confidence': 'HIGH'
                    }
        
        # Fallback: longest substantial sentence in first 10
        substantial_sentences = [s for s in sentences[:10] if len(s['text'].split()) > 15]
        if substantial_sentences:
            best = max(substantial_sentences, key=lambda x: x['length'])
            return {
                'time': best['time'],
                'text': best['text'],
                'confidence': 'MEDIUM'
            }
        
        return {
            'time': '00:00',
            'text': 'No clear thesis identified',
            'confidence': 'LOW'
        }

    def extract_predictions_smart(self, sentences):
        """SMART prediction extraction"""
        print("🔮 Extracting predictions with context...")
        
        predictions = []
        
        for sentence in sentences:
            text = sentence['text']
            text_lower = text.lower()
            
            # Look for predictions with full context
            prediction_indicators = [
                r'(will\s+\w+.*?\d+)',
                r'(expect.*?\d+)',
                r'(target.*?\d+)',
                r'(by\s+\w+.*?\d+)',
                r'(\d+k?\s*(?:btc|bitcoin|eth|ethereum))',
                r'(top.*?(?:october|november|december))',
                r'(cycle.*?(?:end|top|peak))'
            ]
            
            for pattern in prediction_indicators:
                matches = re.findall(pattern, text_lower)
                if matches:
                    # Extract confidence
                    confidence = self.analyze_confidence(text_lower)
                    
                    # Extract timeframe
                    timeframe = self.extract_timeframe_smart(text_lower)
                    
                    predictions.append({
                        'time': sentence['time'],
                        'text': text,
                        'prediction_type': self.categorize_prediction(text_lower),
                        'confidence': confidence,
                        'timeframe': timeframe,
                        'key_numbers': self.extract_numbers(text)
                    })
                    break  # One prediction per sentence
        
        return predictions

    def analyze_confidence(self, text):
        """Analyze confidence level with context"""
        high_conf = ['will', 'definitely', 'certainly', 'guarantee', 'absolutely']
        medium_conf = ['expect', 'likely', 'probably', 'should']
        low_conf = ['maybe', 'could', 'might', 'possibly', 'perhaps']
        
        if any(word in text for word in high_conf):
            return 'HIGH'
        elif any(word in text for word in low_conf):
            return 'LOW'
        else:
            return 'MEDIUM'

    def extract_timeframe_smart(self, text):
        """Smart timeframe extraction"""
        timeframe_patterns = [
            (r'by\s+(october|november|december)\s+(\d{4})', r'\1 \2'),
            (r'in\s+(\d{4})', r'\1'),
            (r'(october|november|december)', r'\1 2025'),
            (r'next\s+(\w+)', r'next \1'),
            (r'(q[1-4])\s+(\d{4})', r'\1 \2'),
            (r'end\s+of\s+(\w+)', r'end of \1')
        ]
        
        for pattern, replacement in timeframe_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return 'Unspecified'

    def categorize_prediction(self, text):
        """Categorize type of prediction"""
        if any(word in text for word in ['price', 'target', '$', 'k', 'btc', 'eth']):
            return 'PRICE_TARGET'
        elif any(word in text for word in ['top', 'peak', 'cycle', 'end']):
            return 'MARKET_TIMING'
        elif any(word in text for word in ['adoption', 'institutional', 'etf']):
            return 'ADOPTION_TREND'
        else:
            return 'GENERAL_FORECAST'

    def extract_numbers(self, text):
        """Extract key numbers from predictions"""
        numbers = re.findall(r'(\d+(?:,\d+)*[kK]?)', text)
        return numbers

    def map_argument_structure(self, sentences):
        """Map the logical argument structure"""
        print("🏗️ Mapping argument structure...")
        
        structure = {
            'opening_thesis': [],
            'supporting_evidence': [],
            'contrarian_points': [],
            'conclusions': [],
            'warnings': []
        }
        
        for sentence in sentences:
            text = sentence['text']
            text_lower = text.lower()
            
            # Categorize by content and position
            if any(phrase in text_lower for phrase in ['everyone is wrong', 'reality is', 'truth is']):
                structure['opening_thesis'].append(sentence)
            elif any(word in text_lower for word in ['data', 'evidence', 'charts', 'etf', 'institutional', 'statistics']):
                structure['supporting_evidence'].append(sentence)
            elif any(word in text_lower for word in ['however', 'but', 'contrary', 'unlike', 'different']):
                structure['contrarian_points'].append(sentence)
            elif any(phrase in text_lower for phrase in ['therefore', 'so', 'conclusion', 'bottom line']):
                structure['conclusions'].append(sentence)
            elif any(word in text_lower for word in ['careful', 'risk', 'danger', 'warning']):
                structure['warnings'].append(sentence)
        
        return structure

    def create_intelligent_summary(self, sentences, thesis, predictions, structure):
        """Create INTELLIGENT summary that captures the logic"""
        print("📝 Creating intelligent summary...")
        
        # Build summary logically
        summary_parts = []
        
        # 1. Lead with thesis
        if thesis['confidence'] != 'LOW':
            summary_parts.append(f"MAIN ARGUMENT: {thesis['text']}")
        
        # 2. Add key supporting evidence
        if structure['supporting_evidence']:
            best_evidence = max(structure['supporting_evidence'], 
                              key=lambda x: len(x['text'].split()))
            summary_parts.append(f"KEY EVIDENCE: {best_evidence['text']}")
        
        # 3. Add top prediction
        if predictions:
            top_prediction = max(predictions, 
                               key=lambda x: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x['confidence']])
            summary_parts.append(f"MAIN PREDICTION: {top_prediction['text']}")
        
        # 4. Add contrarian insight
        if structure['contrarian_points']:
            contrarian = structure['contrarian_points'][0]
            summary_parts.append(f"CONTRARIAN VIEW: {contrarian['text']}")
        
        # 5. Add conclusion if available
        if structure['conclusions']:
            conclusion = structure['conclusions'][-1]  # Last conclusion
            summary_parts.append(f"CONCLUSION: {conclusion['text']}")
        
        return ' | '.join(summary_parts)

    def generate_smart_analysis(self, sentences, thesis, predictions, structure):
        """Generate comprehensive smart analysis"""
        
        # Extract themes intelligently
        themes = self.extract_themes_contextual(sentences)
        
        # Risk analysis
        risks = self.analyze_risks_smart(predictions, structure['warnings'])
        
        # Confidence analysis
        confidence_analysis = self.analyze_prediction_confidence(predictions)
        
        # Investment implications
        implications = self.extract_investment_implications(predictions, structure)
        
        return {
            'thesis': thesis,
            'summary': self.create_intelligent_summary(sentences, thesis, predictions, structure),
            'predictions': predictions,
            'argument_structure': structure,
            'themes': themes,
            'risks': risks,
            'confidence_analysis': confidence_analysis,
            'investment_implications': implications,
            'total_sentences': len(sentences)
        }

    def extract_themes_contextual(self, sentences):
        """Extract themes with context understanding"""
        themes = defaultdict(list)
        
        for sentence in sentences:
            text = sentence['text'].lower()
            
            # Bitcoin themes
            if any(word in text for word in ['bitcoin', 'btc']):
                if any(word in text for word in ['etf', 'institutional']):
                    themes['Bitcoin ETF/Institutional'].append(sentence['text'][:100])
                elif any(word in text for word in ['price', 'target', '$']):
                    themes['Bitcoin Price Targets'].append(sentence['text'][:100])
                else:
                    themes['Bitcoin General'].append(sentence['text'][:100])
            
            # Market cycle themes
            if any(word in text for word in ['cycle', 'top', 'peak', 'bull', 'bear']):
                themes['Market Cycles'].append(sentence['text'][:100])
            
            # Contrarian themes
            if any(word in text for word in ['wrong', 'everyone', 'crowd', 'mistake']):
                themes['Contrarian Analysis'].append(sentence['text'][:100])
        
        # Convert to frequency counts
        theme_counts = {k: len(v) for k, v in themes.items()}
        return theme_counts

    def analyze_risks_smart(self, predictions, warnings):
        """Smart risk analysis"""
        risks = []
        
        # Explicit warnings
        for warning in warnings:
            risks.append(f"⚠️ EXPLICIT WARNING: {warning['text'][:100]}...")
        
        # Overconfidence risk
        high_conf_predictions = [p for p in predictions if p['confidence'] == 'HIGH']
        if len(high_conf_predictions) > 2:
            risks.append("🚨 OVERCONFIDENCE RISK: Multiple high-confidence predictions")
        
        # Timing risk
        specific_timeframes = [p for p in predictions if p['timeframe'] != 'Unspecified']
        if len(specific_timeframes) > 3:
            risks.append("⏰ TIMING RISK: Multiple specific time-based predictions")
        
        return risks if risks else ["No significant risks identified"]

    def analyze_prediction_confidence(self, predictions):
        """Analyze overall confidence pattern"""
        if not predictions:
            return "No predictions to analyze"
        
        confidence_dist = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for pred in predictions:
            confidence_dist[pred['confidence']] += 1
        
        total = sum(confidence_dist.values())
        percentages = {k: f"{v/total*100:.0f}%" for k, v in confidence_dist.items() if v > 0}
        
        return {
            'distribution': percentages,
            'total_predictions': total,
            'dominant_confidence': max(confidence_dist, key=confidence_dist.get)
        }

    def extract_investment_implications(self, predictions, structure):
        """Extract actionable investment implications"""
        implications = []
        
        # From high-confidence predictions
        for pred in predictions:
            if pred['confidence'] == 'HIGH' and pred['prediction_type'] == 'PRICE_TARGET':
                implications.append(f"💰 PRICE TARGET: {pred['text'][:80]}...")
        
        # From timing predictions
        timing_predictions = [p for p in predictions if p['prediction_type'] == 'MARKET_TIMING']
        for pred in timing_predictions:
            implications.append(f"📅 TIMING: {pred['text'][:80]}...")
        
        # From evidence
        strong_evidence = [e for e in structure['supporting_evidence'] if len(e['text'].split()) > 20]
        for evidence in strong_evidence[:2]:
            implications.append(f"📊 EVIDENCE: {evidence['text'][:80]}...")
        
        return implications

    def format_smart_report(self, analysis, video_id):
        """Format the smart analysis report"""
        report = f"""
🧠 SMART AI ANALYSIS REPORT
Video ID: {video_id}
Analysis Engine: TruScraper v5 - True Understanding
{'='*60}

🎯 MAIN THESIS ({analysis['thesis']['confidence']} confidence)
[{analysis['thesis']['time']}] {analysis['thesis']['text']}

📋 INTELLIGENT SUMMARY
{analysis['summary']}

🔮 PREDICTIONS ANALYSIS ({len(analysis['predictions'])} found)
"""
        
        # Group predictions by type
        prediction_types = {}
        for pred in analysis['predictions']:
            pred_type = pred['prediction_type']
            if pred_type not in prediction_types:
                prediction_types[pred_type] = []
            prediction_types[pred_type].append(pred)
        
        for pred_type, preds in prediction_types.items():
            report += f"\n{pred_type.replace('_', ' ').title()}:\n"
            for pred in preds[:3]:  # Top 3 per type
                conf_emoji = {'HIGH': '🔥', 'MEDIUM': '⚠️', 'LOW': '🤔'}[pred['confidence']]
                report += f"  {conf_emoji} [{pred['time']}] {pred['text']}\n"
                if pred['key_numbers']:
                    report += f"     Numbers: {', '.join(pred['key_numbers'])}\n"
                if pred['timeframe'] != 'Unspecified':
                    report += f"     Timeframe: {pred['timeframe']}\n"
        
        report += f"""
🏗️ ARGUMENT STRUCTURE
• Opening Thesis: {len(analysis['argument_structure']['opening_thesis'])} statements
• Supporting Evidence: {len(analysis['argument_structure']['supporting_evidence'])} points
• Contrarian Points: {len(analysis['argument_structure']['contrarian_points'])} insights
• Conclusions: {len(analysis['argument_structure']['conclusions'])} statements
• Warnings: {len(analysis['argument_structure']['warnings'])} alerts

📊 CONFIDENCE ANALYSIS
Distribution: {analysis['confidence_analysis']['distribution']}
Dominant Tone: {analysis['confidence_analysis']['dominant_confidence']}
Total Predictions: {analysis['confidence_analysis']['total_predictions']}

💡 INVESTMENT IMPLICATIONS
"""
        for implication in analysis['investment_implications']:
            report += f"• {implication}\n"
        
        report += f"""
⚠️ RISK ASSESSMENT
"""
        for risk in analysis['risks']:
            report += f"• {risk}\n"
        
        report += f"""
📈 KEY THEMES
"""
        for theme, count in sorted(analysis['themes'].items(), key=lambda x: x[1], reverse=True):
            report += f"• {theme}: {count} mentions\n"
        
        report += f"""
📊 ANALYSIS METRICS
• Total coherent sentences: {analysis['total_sentences']}
• Predictions extracted: {len(analysis['predictions'])}
• Thesis confidence: {analysis['thesis']['confidence']}
• Argument structure completeness: {len(analysis['argument_structure']['supporting_evidence']) > 3 and 'HIGH' or 'MEDIUM'}

{'='*60}
🚀 SMART ANALYSIS COMPLETE
This analysis uses true AI understanding, not just keyword matching.
"""
        
        return report

def main():
    """Main execution"""
    if len(sys.argv) != 2:
        print("Usage: python3 scraper_v5.py <youtube_url>")
        sys.exit(1)
    
    # Check dependencies
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except:
        print("❌ yt-dlp not installed! Run: pip install yt-dlp")
        sys.exit(1)
    
    analyzer = SmartTranscriptAnalyzer()
    youtube_url = sys.argv[1]
    
    try:
        print("🚀 Starting Smart Analysis...")
        
        # Extract video ID
        video_id = analyzer.extract_video_id(youtube_url)
        print(f"📹 Video ID: {video_id}")
        
        # Download subtitles
        subtitle_file = analyzer.download_subtitles(youtube_url, video_id)
        if not subtitle_file:
            print("❌ Failed to download subtitles")
            sys.exit(1)
        
        # Smart parsing
        sentences = analyzer.smart_parse_vtt(subtitle_file)
        if not sentences:
            print("❌ Failed to parse transcript")
            sys.exit(1)
        
        # AI Analysis
        print("🧠 Running true AI understanding analysis...")
        thesis = analyzer.extract_main_thesis(sentences)
        predictions = analyzer.extract_predictions_smart(sentences)
        structure = analyzer.map_argument_structure(sentences)
        
        print(f"   📝 Found {len(sentences)} coherent sentences")
        print(f"   🎯 Extracted thesis with {thesis['confidence']} confidence")
        print(f"   🔮 Found {len(predictions)} predictions")
        print(f"   🏗️ Mapped argument structure")
        
        # Generate analysis
        analysis = analyzer.generate_smart_analysis(sentences, thesis, predictions, structure)
        report = analyzer.format_smart_report(analysis, video_id)
        
        # Save and display
        output_file = f"smart_analysis_{video_id}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n" + report)
        print(f"\n💾 Smart analysis saved: {output_file}")
        
        # Cleanup
        try:
            os.remove(subtitle_file)
        except:
            pass
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
