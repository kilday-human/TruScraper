#!/usr/bin/env python3
"""
TruScraper v7 - INVESTMENT INTELLIGENCE ENGINE
True semantic understanding + audio conviction + actionable investment analysis
This version thinks like a professional financial analyst
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
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

# Audio analysis imports
try:
    import librosa
    import numpy as np
    import whisper
    AUDIO_AVAILABLE = True
    print("🎙️ Audio conviction analysis available")
except ImportError as e:
    AUDIO_AVAILABLE = False
    print(f"📝 Text-only analysis (missing: {e})")

class PredictionType(Enum):
    """Sophisticated prediction categorization"""
    LOCAL_TOP = "local_top"           # Short-term peak (weeks/months)
    CYCLE_TOP = "cycle_top"           # Multi-year cycle peak
    PRICE_TARGET = "price_target"     # Specific price level
    TIMEFRAME = "timeframe"           # When something will happen
    INSTITUTIONAL = "institutional"   # Adoption/ETF trends
    FUNDAMENTAL = "fundamental"       # Market structure changes

class ConvictionLevel(Enum):
    """Conviction levels with investment implications"""
    EXTREME = "extreme"     # Bet the farm
    HIGH = "high"          # Strong position
    MEDIUM = "medium"      # Moderate allocation
    LOW = "low"           # Small position
    UNCERTAIN = "uncertain" # Wait and see

@dataclass
class InvestmentSignal:
    """Structured investment recommendation"""
    action: str                    # BUY, SELL, HOLD, WAIT
    asset: str                     # BTC, ETH, etc.
    conviction: ConvictionLevel    # How strong the signal is
    timeframe: str                 # When to act
    rationale: str                 # Why this recommendation
    risk_level: str                # HIGH, MEDIUM, LOW
    position_size: str             # LARGE, MEDIUM, SMALL
    audio_conviction: Optional[float] = None  # Voice conviction 0-1

@dataclass
class SemanticConcept:
    """Crypto/finance concept with context"""
    concept: str                   # "asset maturation"
    definition: str                # What it means
    implications: List[str]        # Investment implications
    supporting_evidence: List[str] # Supporting points
    confidence: float              # How certain we are

class CryptoSemanticEngine:
    """Understands crypto concepts and relationships"""
    
    def __init__(self):
        print("🧠 Initializing Crypto Semantic Engine...")
        
        # Core crypto concept definitions
        self.concept_definitions = {
            'asset_maturation': {
                'patterns': ['maturation', 'mainstream', 'institutional adoption', 'etf'],
                'meaning': 'Crypto transitioning from speculative to traditional asset class',
                'bullish_implications': ['Longer cycles', 'Higher sustained prices', 'Less volatility'],
                'investment_signals': ['Increase allocation', 'Longer hold periods', 'Institutional following']
            },
            'cycle_extension': {
                'patterns': ['extended cycle', 'longer cycle', 'different this time', 'new paradigm'],
                'meaning': 'Traditional 4-year cycles may be changing due to institutional adoption',
                'bullish_implications': ['Higher peaks', 'Delayed tops', 'Multiple legs up'],
                'investment_signals': ['Hold through traditional cycle tops', 'Scale out higher']
            },
            'local_vs_cycle_top': {
                'patterns': ['local top', 'local peak', 'intermediate high'],
                'meaning': 'Temporary peak before continuation higher',
                'bullish_implications': ['Buying opportunity on pullback', 'Trend continuation expected'],
                'investment_signals': ['Buy dips', 'Add on weakness', 'Maintain positions']
            },
            'institutional_bid': {
                'patterns': ['institutional', 'wall street', 'etf', 'blackrock', 'corporate treasuries'],
                'meaning': 'Large institutions actively buying and holding crypto',
                'bullish_implications': ['Price floor support', 'Reduced selling pressure', 'Legitimacy'],
                'investment_signals': ['Strong foundation for higher prices', 'Lower risk']
            },
            'market_psychology': {
                'patterns': ['everyone is wrong', 'crowd is wrong', 'contrarian', 'consensus'],
                'meaning': 'Positioning against popular sentiment',
                'bullish_implications': ['Opportunity when others are fearful', 'Less competition'],
                'investment_signals': ['Counter-trend positioning', 'Fade the crowd']
            }
        }
        
        # Price level context
        self.price_context = {
            'bitcoin': {
                'current_range': (90000, 110000),  # Approximate current levels
                'psychological_levels': [100000, 150000, 200000],
                'technical_levels': [120000, 150000, 180000]
            },
            'ethereum': {
                'current_range': (3000, 4000),
                'psychological_levels': [5000, 7000, 10000],
                'technical_levels': [4500, 6000, 8000]
            }
        }
    
    def identify_concepts(self, text: str) -> List[SemanticConcept]:
        """Identify crypto concepts in text with full context"""
        concepts = []
        text_lower = text.lower()
        
        for concept_name, concept_data in self.concept_definitions.items():
            # Check if concept is present
            pattern_matches = sum(1 for pattern in concept_data['patterns'] if pattern in text_lower)
            
            if pattern_matches > 0:
                confidence = min(1.0, pattern_matches / len(concept_data['patterns']))
                
                concept = SemanticConcept(
                    concept=concept_name,
                    definition=concept_data['meaning'],
                    implications=concept_data['bullish_implications'],
                    supporting_evidence=[text],
                    confidence=confidence
                )
                concepts.append(concept)
        
        return concepts
    
    def extract_price_targets(self, text: str) -> List[Dict]:
        """Extract price targets with semantic understanding"""
        targets = []
        
        # Bitcoin targets
        btc_patterns = [
            r'(\d+k?)\s*(?:btc|bitcoin)',
            r'bitcoin.*?(\d+k?)',
            r'btc.*?(\d+k?)'
        ]
        
        for pattern in btc_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    # Convert to numeric
                    if 'k' in match:
                        price = int(match.replace('k', '')) * 1000
                    else:
                        price = int(match)
                    
                    # Context analysis
                    context = self.analyze_price_context('bitcoin', price, text)
                    
                    targets.append({
                        'asset': 'BTC',
                        'price': price,
                        'raw_text': match,
                        'context': context,
                        'significance': self.assess_price_significance('bitcoin', price)
                    })
                except ValueError:
                    continue
        
        # Ethereum targets
        eth_patterns = [
            r'(\d+k?)\s*(?:eth|ethereum)',
            r'ethereum.*?(\d+k?)',
            r'eth.*?(\d+k?)'
        ]
        
        for pattern in eth_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    if 'k' in match:
                        price = int(match.replace('k', '')) * 1000
                    else:
                        price = int(match)
                    
                    context = self.analyze_price_context('ethereum', price, text)
                    
                    targets.append({
                        'asset': 'ETH',
                        'price': price,
                        'raw_text': match,
                        'context': context,
                        'significance': self.assess_price_significance('ethereum', price)
                    })
                except ValueError:
                    continue
        
        return targets
    
    def analyze_price_context(self, asset: str, price: int, text: str) -> Dict:
        """Analyze price target context"""
        context = {
            'is_local_top': 'local' in text.lower() or 'intermediate' in text.lower(),
            'is_cycle_top': 'cycle' in text.lower() and 'top' in text.lower(),
            'is_conservative': False,
            'is_aggressive': False,
            'timeframe_urgency': 'medium'
        }
        
        if asset in self.price_context:
            asset_data = self.price_context[asset]
            current_min, current_max = asset_data['current_range']
            
            # Determine if target is conservative or aggressive
            multiplier = price / current_max
            
            if multiplier < 1.5:
                context['is_conservative'] = True
            elif multiplier > 3.0:
                context['is_aggressive'] = True
            
            # Check if it's near psychological levels
            for psych_level in asset_data['psychological_levels']:
                if abs(price - psych_level) / psych_level < 0.1:  # Within 10%
                    context['near_psychological_level'] = psych_level
        
        return context
    
    def assess_price_significance(self, asset: str, price: int) -> str:
        """Assess how significant a price target is"""
        if asset in self.price_context:
            current_max = self.price_context[asset]['current_range'][1]
            multiplier = price / current_max
            
            if multiplier > 5:
                return 'EXTREME'
            elif multiplier > 3:
                return 'VERY_HIGH'
            elif multiplier > 2:
                return 'HIGH'
            elif multiplier > 1.5:
                return 'MODERATE'
            else:
                return 'LOW'
        
        return 'UNKNOWN'

class ArgumentFlowMapper:
    """Maps logical argument structure for investment analysis"""
    
    def __init__(self):
        self.argument_indicators = {
            'thesis_statements': [
                'everyone is wrong', 'reality is', 'key point', 'main argument',
                'thesis', 'central claim', 'fundamental point'
            ],
            'evidence_markers': [
                'data shows', 'evidence suggests', 'charts indicate', 'statistics reveal',
                'etf flows', 'institutional buying', 'adoption metrics'
            ],
            'reasoning_connectors': [
                'because', 'since', 'therefore', 'as a result', 'this means',
                'which suggests', 'leading to', 'consequently'
            ],
            'contrarian_signals': [
                'however', 'but', 'contrary to', 'unlike', 'opposite',
                'crowd thinks', 'consensus believes', 'most people'
            ],
            'conclusion_markers': [
                'therefore', 'so', 'bottom line', 'conclusion', 'result',
                'this leads to', 'ultimately', 'final point'
            ]
        }
    
    def map_argument_flow(self, sentences: List[Dict]) -> Dict:
        """Map the complete argument structure"""
        flow = {
            'thesis': None,
            'supporting_evidence': [],
            'reasoning_chain': [],
            'contrarian_points': [],
            'conclusions': [],
            'investment_implications': []
        }
        
        for i, sentence in enumerate(sentences):
            text = sentence['text']
            text_lower = text.lower()
            
            # Identify thesis
            if not flow['thesis'] and any(indicator in text_lower for indicator in self.argument_indicators['thesis_statements']):
                flow['thesis'] = {
                    'text': text,
                    'position': i,
                    'confidence': self.assess_statement_confidence(text)
                }
            
            # Identify evidence
            if any(marker in text_lower for marker in self.argument_indicators['evidence_markers']):
                flow['supporting_evidence'].append({
                    'text': text,
                    'position': i,
                    'evidence_type': self.categorize_evidence(text),
                    'strength': self.assess_evidence_strength(text)
                })
            
            # Identify reasoning
            if any(connector in text_lower for connector in self.argument_indicators['reasoning_connectors']):
                flow['reasoning_chain'].append({
                    'text': text,
                    'position': i,
                    'logic_type': self.categorize_logic(text)
                })
            
            # Identify contrarian points
            if any(signal in text_lower for signal in self.argument_indicators['contrarian_signals']):
                flow['contrarian_points'].append({
                    'text': text,
                    'position': i,
                    'contrarian_strength': self.assess_contrarian_strength(text)
                })
            
            # Identify conclusions
            if any(marker in text_lower for marker in self.argument_indicators['conclusion_markers']):
                flow['conclusions'].append({
                    'text': text,
                    'position': i,
                    'conclusion_type': self.categorize_conclusion(text)
                })
        
        return flow
    
    def assess_statement_confidence(self, text: str) -> float:
        """Assess how confident a statement sounds"""
        confidence_words = ['definitely', 'certainly', 'absolutely', 'guarantee', 'sure']
        uncertainty_words = ['maybe', 'possibly', 'could', 'might', 'perhaps']
        
        text_lower = text.lower()
        conf_count = sum(1 for word in confidence_words if word in text_lower)
        uncertain_count = sum(1 for word in uncertainty_words if word in text_lower)
        
        base_confidence = 0.5
        base_confidence += conf_count * 0.2
        base_confidence -= uncertain_count * 0.2
        
        return max(0.0, min(1.0, base_confidence))
    
    def categorize_evidence(self, text: str) -> str:
        """Categorize type of evidence presented"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['data', 'statistics', 'numbers', 'metrics']):
            return 'QUANTITATIVE'
        elif any(word in text_lower for word in ['etf', 'institutional', 'blackrock', 'wall street']):
            return 'INSTITUTIONAL'
        elif any(word in text_lower for word in ['history', 'previous', 'past', 'cycle']):
            return 'HISTORICAL'
        elif any(word in text_lower for word in ['chart', 'technical', 'price', 'level']):
            return 'TECHNICAL'
        else:
            return 'ANECDOTAL'
    
    def assess_evidence_strength(self, text: str) -> str:
        """Assess strength of evidence"""
        strong_indicators = ['data shows', 'statistics prove', 'evidence confirms']
        moderate_indicators = ['suggests', 'indicates', 'points to']
        weak_indicators = ['seems', 'appears', 'might suggest']
        
        text_lower = text.lower()
        
        if any(indicator in text_lower for indicator in strong_indicators):
            return 'STRONG'
        elif any(indicator in text_lower for indicator in moderate_indicators):
            return 'MODERATE'
        elif any(indicator in text_lower for indicator in weak_indicators):
            return 'WEAK'
        else:
            return 'UNCLEAR'
    
    def categorize_logic(self, text: str) -> str:
        """Categorize type of logical reasoning"""
        text_lower = text.lower()
        
        if 'because' in text_lower or 'since' in text_lower:
            return 'CAUSAL'
        elif 'therefore' in text_lower or 'as a result' in text_lower:
            return 'CONSEQUENTIAL'
        elif 'this means' in text_lower or 'which suggests' in text_lower:
            return 'INFERENTIAL'
        else:
            return 'GENERAL'
    
    def assess_contrarian_strength(self, text: str) -> float:
        """Assess how contrarian a viewpoint is"""
        strong_contrarian = ['everyone is wrong', 'crowd is mistaken', 'consensus is false']
        moderate_contrarian = ['however', 'but', 'contrary to']
        
        text_lower = text.lower()
        
        if any(phrase in text_lower for phrase in strong_contrarian):
            return 1.0
        elif any(phrase in text_lower for phrase in moderate_contrarian):
            return 0.6
        else:
            return 0.3
    
    def categorize_conclusion(self, text: str) -> str:
        """Categorize type of conclusion"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['buy', 'sell', 'hold', 'invest']):
            return 'ACTIONABLE'
        elif any(word in text_lower for word in ['price', 'target', 'level']):
            return 'PREDICTIVE'
        elif any(word in text_lower for word in ['therefore', 'ultimately', 'final']):
            return 'SUMMARY'
        else:
            return 'GENERAL'

class InvestmentDecisionEngine:
    """Generates actionable investment recommendations"""
    
    def __init__(self):
        self.risk_tolerance = 'moderate'  # Can be configured
        
        # Decision frameworks
        self.conviction_thresholds = {
            ConvictionLevel.EXTREME: 0.9,
            ConvictionLevel.HIGH: 0.7,
            ConvictionLevel.MEDIUM: 0.5,
            ConvictionLevel.LOW: 0.3,
            ConvictionLevel.UNCERTAIN: 0.0
        }
        
        # Position sizing based on conviction
        self.position_sizes = {
            ConvictionLevel.EXTREME: 'LARGE (20-30%)',
            ConvictionLevel.HIGH: 'MEDIUM (10-20%)',
            ConvictionLevel.MEDIUM: 'SMALL (5-10%)',
            ConvictionLevel.LOW: 'MINIMAL (1-5%)',
            ConvictionLevel.UNCERTAIN: 'NONE (0%)'
        }
    
    def generate_investment_signals(self, analysis_data: Dict) -> List[InvestmentSignal]:
        """Generate concrete investment recommendations"""
        signals = []
        
        concepts = analysis_data.get('concepts', [])
        price_targets = analysis_data.get('price_targets', [])
        argument_flow = analysis_data.get('argument_flow', {})
        audio_conviction = analysis_data.get('audio_conviction', {})
        
        # Process each price target into investment signal
        for target in price_targets:
            signal = self.create_price_target_signal(target, concepts, argument_flow, audio_conviction)
            if signal:
                signals.append(signal)
        
        # Generate strategic signals from concepts
        for concept in concepts:
            strategic_signal = self.create_strategic_signal(concept, argument_flow)
            if strategic_signal:
                signals.append(strategic_signal)
        
        # Sort by conviction level
        signals.sort(key=lambda x: list(ConvictionLevel).index(x.conviction))
        
        return signals
    
    def create_price_target_signal(self, target: Dict, concepts: List, argument_flow: Dict, audio_data: Dict) -> Optional[InvestmentSignal]:
        """Create investment signal from price target"""
        asset = target['asset']
        price = target['price']
        context = target['context']
        significance = target['significance']
        
        # Determine conviction level
        base_conviction = 0.5
        
        # Adjust for price significance
        significance_boost = {
            'EXTREME': 0.3, 'VERY_HIGH': 0.2, 'HIGH': 0.15,
            'MODERATE': 0.1, 'LOW': 0.05, 'UNKNOWN': 0.0
        }
        base_conviction += significance_boost.get(significance, 0.0)
        
        # Adjust for context
        if context.get('is_local_top'):
            # Local top = buying opportunity on pullback
            action = 'BUY_DIP'
            base_conviction += 0.1
        elif context.get('is_cycle_top'):
            # Cycle top = take profits
            action = 'TAKE_PROFITS'
            base_conviction += 0.05
        else:
            action = 'BUY'
        
        # Adjust for supporting concepts
        for concept in concepts:
            if concept.concept in ['asset_maturation', 'institutional_bid']:
                base_conviction += 0.1
            elif concept.concept == 'cycle_extension':
                base_conviction += 0.15
        
        # Add audio conviction if available
        audio_boost = 0.0
        if audio_data and 'conviction_results' in audio_data:
            for result in audio_data['conviction_results']:
                if str(price) in result['prediction']['text']:
                    audio_boost = result['audio_conviction']['confidence_score'] * 0.2
                    break
        
        final_conviction = min(1.0, base_conviction + audio_boost)
        
        # Convert to conviction level
        conviction_level = self.map_conviction_level(final_conviction)
        
        # Determine timeframe
        timeframe = self.extract_timeframe(target, argument_flow)
        
        # Generate rationale
        rationale = self.generate_rationale(target, concepts, context, audio_boost > 0)
        
        return InvestmentSignal(
            action=action,
            asset=asset,
            conviction=conviction_level,
            timeframe=timeframe,
            rationale=rationale,
            risk_level=self.assess_risk_level(significance, context),
            position_size=self.position_sizes[conviction_level],
            audio_conviction=audio_boost if audio_boost > 0 else None
        )
    
    def create_strategic_signal(self, concept: SemanticConcept, argument_flow: Dict) -> Optional[InvestmentSignal]:
        """Create strategic investment signal from concept"""
        if concept.concept == 'asset_maturation':
            return InvestmentSignal(
                action='INCREASE_ALLOCATION',
                asset='BTC/ETH',
                conviction=ConvictionLevel.HIGH,
                timeframe='LONG_TERM',
                rationale='Asset maturation suggests longer cycles and institutional adoption',
                risk_level='MEDIUM',
                position_size='MEDIUM (10-20%)'
            )
        elif concept.concept == 'cycle_extension':
            return InvestmentSignal(
                action='HOLD_THROUGH_RESISTANCE',
                asset='BTC/ETH', 
                conviction=ConvictionLevel.MEDIUM,
                timeframe='6-12_MONTHS',
                rationale='Extended cycle theory suggests traditional tops may be false signals',
                risk_level='MEDIUM',
                position_size='MAINTAIN_CURRENT'
            )
        
        return None
    
    def map_conviction_level(self, conviction_score: float) -> ConvictionLevel:
        """Map numeric conviction to enum"""
        if conviction_score >= 0.9:
            return ConvictionLevel.EXTREME
        elif conviction_score >= 0.7:
            return ConvictionLevel.HIGH
        elif conviction_score >= 0.5:
            return ConvictionLevel.MEDIUM
        elif conviction_score >= 0.3:
            return ConvictionLevel.LOW
        else:
            return ConvictionLevel.UNCERTAIN
    
    def extract_timeframe(self, target: Dict, argument_flow: Dict) -> str:
        """Extract investment timeframe"""
        context = target['context']
        
        if context.get('is_local_top'):
            return '1-3_MONTHS'
        elif context.get('is_cycle_top'):
            return '6-18_MONTHS'
        else:
            return '3-12_MONTHS'
    
    def assess_risk_level(self, significance: str, context: Dict) -> str:
        """Assess risk level of recommendation"""
        if significance in ['EXTREME', 'VERY_HIGH']:
            return 'HIGH'
        elif context.get('is_aggressive'):
            return 'HIGH'
        elif context.get('is_conservative'):
            return 'LOW'
        else:
            return 'MEDIUM'
    
    def generate_rationale(self, target: Dict, concepts: List, context: Dict, has_audio: bool) -> str:
        """Generate investment rationale"""
        rationale_parts = []
        
        # Base rationale
        price = target['price']
        asset = target['asset']
        rationale_parts.append(f"{asset} target of ${price:,}")
        
        # Context rationale
        if context.get('is_local_top'):
            rationale_parts.append("positioned as local top (buying opportunity on pullback)")
        elif context.get('is_cycle_top'):
            rationale_parts.append("positioned as cycle peak (profit-taking level)")
        
        # Concept support
        concept_support = []
        for concept in concepts:
            if concept.confidence > 0.5:
                concept_support.append(concept.concept.replace('_', ' '))
        
        if concept_support:
            rationale_parts.append(f"supported by {', '.join(concept_support)}")
        
        # Audio conviction
        if has_audio:
            rationale_parts.append("high voice conviction detected")
        
        return '; '.join(rationale_parts)

class CoherentNarrativeSynthesizer:
    """Creates coherent, logical summaries instead of fragments"""
    
    def __init__(self):
        pass
    
    def synthesize_narrative(self, analysis_data: Dict) -> Dict:
        """Create coherent narrative from analysis"""
        concepts = analysis_data.get('concepts', [])
        argument_flow = analysis_data.get('argument_flow', {})
        price_targets = analysis_data.get('price_targets', [])
        investment_signals = analysis_data.get('investment_signals', [])
        
        narrative = {
            'executive_summary': self.create_executive_summary(concepts, argument_flow, investment_signals),
            'main_thesis': self.extract_main_thesis(argument_flow),
            'investment_case': self.build_investment_case(concepts, price_targets, investment_signals),
            'risk_assessment': self.assess_risks(investment_signals, concepts),
            'action_plan': self.create_action_plan(investment_signals)
        }
        
        return narrative
    
    def create_executive_summary(self, concepts: List, argument_flow: Dict, signals: List) -> str:
        """Create coherent executive summary"""
        summary_parts = []
        
        # Main thesis
        if argument_flow.get('thesis'):
            thesis_text = argument_flow['thesis']['text']
            # Clean up the thesis
            clean_thesis = self.clean_statement(thesis_text)
            summary_parts.append(f"THESIS: {clean_thesis}")
        
        # Key concept
        if concepts:
            primary_concept = max(concepts, key=lambda x: x.confidence)
            summary_parts.append(f"KEY INSIGHT: {primary_concept.definition}")
        
        # Top investment signal
        if signals:
            top_signal = signals[0]  # Highest conviction
            summary_parts.append(f"PRIMARY RECOMMENDATION: {top_signal.action} {top_signal.asset} with {top_signal.conviction.value} conviction")
        
        return ' | '.join(summary_parts)
    
    def extract_main_thesis(self, argument_flow: Dict) -> str:
        """Extract and clean main thesis"""
        if argument_flow.get('thesis'):
            raw_thesis = argument_flow['thesis']['text']
            return self.clean_statement(raw_thesis)
        return "No clear thesis identified"
    
    def clean_statement(self, statement: str) -> str:
        """Clean fragmented statements into coherent text"""
        # Remove obvious duplicates and fragments
        words = statement.split()
        cleaned_words = []
        
        # Remove repetitive patterns
        for i, word in enumerate(words):
            # Skip if this word and next few words repeat earlier in sentence
            if i > 5:  # Don't check first few words
                current_phrase = ' '.join(words[i:i+3])
                earlier_text = ' '.join(words[:i])
                if current_phrase.lower() in earlier_text.lower():
                    continue
            
            cleaned_words.append(word)
        
        # Reconstruct sentence
        cleaned = ' '.join(cleaned_words)
        
        # Fix common issues
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces
        cleaned = re.sub(r'(\w+)\s+\1\b', r'\1', cleaned)  # Word repetitions
        
        return cleaned.strip()
    
    def build_investment_case(self, concepts: List, price_targets: List, signals: List) -> str:
        """Build coherent investment case"""
        case_parts = []
        
        # Fundamental case from concepts
        bullish_concepts = [c for c in concepts if c.confidence > 0.5]
        if bullish_concepts:
            concept_names = [c.concept.replace('_', ' ').title() for c in bullish_concepts]
            case_parts.append(f"FUNDAMENTAL DRIVERS: {', '.join(concept_names)}")
        
        # Price targets
        if price_targets:
            btc_targets = [t for t in price_targets if t['asset'] == 'BTC']
            eth_targets = [t for t in price_targets if t['asset'] == 'ETH']
            
            target_text = []
            if btc_targets:
                btc_price = max(btc_targets, key=lambda x: x['price'])['price']
                target_text.append(f"BTC ${btc_price:,}")
            if eth_targets:
                eth_price = max(eth_targets, key=lambda x: x['price'])['price']
                target_text.append(f"ETH ${eth_price:,}")
            
            if target_text:
                case_parts.append(f"PRICE TARGETS: {', '.join(target_text)}")
        
        # Timeline
        high_conviction_signals = [s for s in signals if s.conviction in [ConvictionLevel.HIGH, ConvictionLevel.EXTREME]]
        if high_conviction_signals:
            timeframes = set(s.timeframe for s in high_conviction_signals)
            case_parts.append(f"TIMEFRAME: {', '.join(timeframes)}")
        
        return ' | '.join(case_parts)
    
    def assess_risks(self, signals: List, concepts: List) -> str:
        """Assess investment risks"""
        risks = []
        
        # High conviction risk
        extreme_signals = [s for s in signals if s.conviction == ConvictionLevel.EXTREME]
        if len(extreme_signals) > 1:
            risks.append("OVERCONFIDENCE: Multiple extreme conviction calls detected")
        
        # Timing risk
        short_term_signals = [s for s in signals if '1-3' in s.timeframe]
        if len(short_term_signals) > 2:
            risks.append("TIMING RISK: Multiple short-term predictions")
        
        # Contrarian risk
        contrarian_concepts = [c for c in concepts if 'contrarian' in c.concept]
        if contrarian_concepts:
            risks.append("CONTRARIAN RISK: Betting against consensus")
        
        return '; '.join(risks) if risks else "No significant risks identified"
    
    def create_action_plan(self, signals: List) -> str:
        """Create concrete action plan"""
        if not signals:
            return "No actionable signals identified"
        
        # Group by timeframe
        immediate = [s for s in signals if '1-3' in s.timeframe]
        medium_term = [s for s in signals if '3-12' in s.timeframe or '6-18' in s.timeframe]
        long_term = [s for s in signals if 'LONG' in s.timeframe]
        
        action_parts = []
        
        if immediate:
            top_immediate = max(immediate, key=lambda x: list(ConvictionLevel).index(x.conviction))
            action_parts.append(f"IMMEDIATE: {top_immediate.action} {top_immediate.asset}")
        
        if medium_term:
            top_medium = max(medium_term, key=lambda x: list(ConvictionLevel).index(x.conviction))
            action_parts.append(f"MEDIUM-TERM: {top_medium.action} {top_medium.asset}")
        
        if long_term:
            top_long = max(long_term, key=lambda x: list(ConvictionLevel).index(x.conviction))
            action_parts.append(f"LONG-TERM: {top_long.action} {top_long.asset}")
        
        return ' | '.join(action_parts)

class AudioConvictionAnalyzer:
    """Enhanced audio analysis with investment focus"""
    
    def __init__(self):
        self.whisper_model = None
        self.sample_rate = 22050
        
    def initialize_whisper(self):
        """Initialize Whisper model"""
        if not AUDIO_AVAILABLE:
            return False
        
        try:
            if self.whisper_model is None:
                print("🎯 Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
                print("✅ Whisper ready")
            return True
        except Exception as e:
            print(f"❌ Whisper failed: {e}")
            return False
    
    def analyze_prediction_conviction(self, audio_file: str, predictions: List[Dict]) -> Dict:
        """Analyze voice conviction for specific predictions"""
        if not self.initialize_whisper():
            return {}
        
        try:
            print("🎙️ Analyzing voice conviction for predictions...")
            
            # Transcribe with word timestamps
            result = self.whisper_model.transcribe(audio_file, word_timestamps=True)
            
            conviction_results = []
            
            for pred in predictions:
                # Find matching audio segments
                pred_conviction = self.find_prediction_audio_match(result, pred, audio_file)
                if pred_conviction:
                    conviction_results.append({
                        'prediction': pred,
                        'audio_conviction': pred_conviction
                    })
            
            return {
                'conviction_results': conviction_results,
                'overall_speaker_profile': self.analyze_speaker_profile(result, audio_file)
            }
            
        except Exception as e:
            print(f"❌ Audio conviction analysis error: {e}")
            return {}
    
    def find_prediction_audio_match(self, transcription: Dict, prediction: Dict, audio_file: str) -> Optional[Dict]:
        """Find audio segment matching prediction and analyze conviction"""
        pred_text = prediction['text'].lower()
        pred_time = self.parse_timestamp(prediction['time'])
        
        # Look for matching segments
        for segment in transcription['segments']:
            if abs(segment['start'] - pred_time) < 10.0:  # Within 10 seconds
                return self.analyze_segment_conviction(audio_file, segment, pred_text)
        
        return None
    
    def analyze_segment_conviction(self, audio_file: str, segment: Dict, prediction_text: str) -> Dict:
        """Analyze conviction for specific audio segment"""
        try:
            # Load audio segment
            y, sr = librosa.load(
                audio_file, 
                sr=self.sample_rate,
                offset=segment['start'],
                duration=min(segment['end'] - segment['start'], 15.0)
            )
            
            if len(y) == 0:
                return {}
            
            # Audio features
            rms_energy = float(np.mean(librosa.feature.rms(y=y)[0]))
            
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            pitch_variance = float(np.var(pitch_values)) if pitch_values else 0.0
            
            # Speaking rate (rough estimation)
            tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
            
            # Investment-focused conviction scoring
            conviction_score = self.calculate_investment_conviction(
                rms_energy, pitch_variance, tempo, segment['text'], prediction_text
            )
            
            return {
                'confidence_score': conviction_score,
                'volume_level': rms_energy,
                'pitch_variance': pitch_variance,
                'speaking_tempo': float(tempo),
                'conviction_category': self.categorize_conviction(conviction_score),
                'investment_weight': self.calculate_investment_weight(conviction_score, prediction_text)
            }
            
        except Exception as e:
            print(f"⚠️ Segment conviction analysis error: {e}")
            return {}
    
    def calculate_investment_conviction(self, volume: float, pitch_var: float, tempo: float, 
                                     segment_text: str, prediction_text: str) -> float:
        """Calculate conviction score focused on investment predictions"""
        base_score = 0.5
        
        # Volume confidence (normalized)
        volume_score = min(1.0, volume * 25)  # Adjust multiplier based on typical levels
        base_score += (volume_score - 0.5) * 0.3
        
        # Pitch variance (excitement/emphasis)
        if pitch_var > 1000:  # High variance = emphasis
            base_score += 0.2
        elif pitch_var < 100:  # Very stable = confident delivery
            base_score += 0.1
        
        # Speaking tempo
        if 100 < tempo < 140:  # Confident pace
            base_score += 0.1
        elif tempo > 160:  # Very fast = excited
            base_score += 0.15
        
        # Text analysis for financial conviction
        text_lower = segment_text.lower()
        
        # High conviction financial words
        if any(word in text_lower for word in ['will', 'definitely', 'absolutely', 'guarantee']):
            base_score += 0.2
        
        # Uncertainty words
        if any(word in text_lower for word in ['maybe', 'possibly', 'could', 'might']):
            base_score -= 0.2
        
        # Price/number emphasis
        if any(char in prediction_text for char in ['$', 'k', '000']):
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def categorize_conviction(self, score: float) -> str:
        """Categorize conviction level"""
        if score >= 0.8:
            return 'EXTREME_CONVICTION'
        elif score >= 0.65:
            return 'HIGH_CONVICTION'
        elif score >= 0.45:
            return 'MODERATE_CONVICTION'
        elif score >= 0.25:
            return 'LOW_CONVICTION'
        else:
            return 'UNCERTAIN'
    
    def calculate_investment_weight(self, conviction: float, prediction_text: str) -> float:
        """Calculate how much weight to give this prediction for investment decisions"""
        base_weight = conviction
        
        # Boost for specific price targets
        if any(char in prediction_text for char in ['$', 'k']):
            base_weight += 0.1
        
        # Boost for timeframe specificity
        if any(word in prediction_text.lower() for word in ['october', 'november', 'december', 'q4']):
            base_weight += 0.1
        
        return min(1.0, base_weight)
    
    def analyze_speaker_profile(self, transcription: Dict, audio_file: str) -> Dict:
        """Analyze overall speaker conviction profile"""
        try:
            # Analyze first few segments for overall profile
            segments_to_analyze = transcription['segments'][:10]  # First 10 segments
            
            conviction_scores = []
            for segment in segments_to_analyze:
                try:
                    y, sr = librosa.load(
                        audio_file, 
                        sr=self.sample_rate,
                        offset=segment['start'],
                        duration=min(segment['end'] - segment['start'], 10.0)
                    )
                    
                    if len(y) > 0:
                        rms = float(np.mean(librosa.feature.rms(y=y)[0]))
                        conviction_scores.append(rms)
                except:
                    continue
            
            if conviction_scores:
                avg_conviction = np.mean(conviction_scores)
                conviction_consistency = 1.0 - np.std(conviction_scores)
                
                return {
                    'average_conviction': float(avg_conviction),
                    'conviction_consistency': float(max(0.0, conviction_consistency)),
                    'speaker_type': self.classify_speaker_type(avg_conviction, conviction_consistency)
                }
            
        except Exception as e:
            print(f"⚠️ Speaker profile error: {e}")
        
        return {'speaker_type': 'UNKNOWN'}
    
    def classify_speaker_type(self, avg_conviction: float, consistency: float) -> str:
        """Classify speaker type for investment context"""
        if avg_conviction > 0.03 and consistency > 0.8:
            return 'CONFIDENT_ANALYST'
        elif avg_conviction > 0.03 and consistency < 0.5:
            return 'PASSIONATE_PROMOTER'
        elif avg_conviction < 0.02 and consistency > 0.8:
            return 'CAUTIOUS_COMMENTATOR'
        else:
            return 'UNCERTAIN_SPEAKER'
    
    def parse_timestamp(self, timestamp: str) -> float:
        """Parse MM:SS timestamp to seconds"""
        try:
            parts = timestamp.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            return 0.0
        except:
            return 0.0

class TruScraperV7:
    """Ultimate Investment Intelligence Engine"""
    
    def __init__(self):
        print("🚀 TruScraper v7 - INVESTMENT INTELLIGENCE ENGINE")
        
        # Initialize components
        self.semantic_engine = CryptoSemanticEngine()
        self.argument_mapper = ArgumentFlowMapper()
        self.decision_engine = InvestmentDecisionEngine()
        self.narrative_synthesizer = CoherentNarrativeSynthesizer()
        self.audio_analyzer = AudioConvictionAnalyzer() if AUDIO_AVAILABLE else None
        
        print("✅ All intelligence engines loaded")
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID"""
        match = re.search(r'(?:v=|youtu\.be/)([^&\n?#]+)', url)
        if match:
            return match.group(1)
        raise ValueError("Could not extract video ID")
    
    def download_and_parse_subtitles(self, video_url: str, video_id: str) -> List[Dict]:
        """Download and parse subtitles with sponsor filtering"""
        try:
            # Download subtitles
            print("📥 Downloading subtitles...")
            cmd = [
                'yt-dlp', '--write-auto-subs', '--sub-langs', 'en',
                '--sub-format', 'vtt', '--skip-download',
                '--output', f'transcript_{video_id}.%(ext)s', video_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            subtitle_file = None
            if result.returncode == 0:
                for suffix in ['.en.vtt', '.en-US.vtt', '.en-GB.vtt']:
                    filename = f'transcript_{video_id}{suffix}'
                    if os.path.exists(filename):
                        subtitle_file = filename
                        break
            
            if not subtitle_file:
                return []
            
            # Parse VTT file
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            segments = []
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
                            segments.append({
                                'time': self.format_timestamp(timestamp),
                                'text': text
                            })
            
            # Clean up
            try:
                os.remove(subtitle_file)
            except:
                pass
            
            return segments
            
        except Exception as e:
            print(f"❌ Subtitle processing error: {e}")
            return []
    
    def format_timestamp(self, timestamp: str) -> str:
        """Format timestamp"""
        try:
            parts = timestamp.split(':')
            if len(parts) >= 3:
                h, m, s = parts[0], parts[1], parts[2].split('.')[0]
                return f"{m}:{s}" if h == '00' else f"{h}:{m}:{s}"
        except:
            pass
        return timestamp
    
    def download_audio_for_conviction(self, video_url: str, video_id: str) -> Optional[str]:
        """Download audio for conviction analysis"""
        if not AUDIO_AVAILABLE:
            return None
        
        try:
            print("🎵 Downloading audio for conviction analysis...")
            audio_file = f"audio_{video_id}.wav"
            
            cmd = [
                'yt-dlp', '-x', '--audio-format', 'wav',
                '--audio-quality', '0', '-o', f'audio_{video_id}.%(ext)s',
                video_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(audio_file):
                return audio_file
            
        except Exception as e:
            print(f"❌ Audio download error: {e}")
        
        return None
    
    def run_comprehensive_analysis(self, segments: List[Dict], audio_file: Optional[str] = None) -> Dict:
        """Run comprehensive investment intelligence analysis"""
        print("🧠 Running comprehensive investment intelligence analysis...")
        
        # Combine segments into full text for concept analysis
        full_text = ' '.join([seg['text'] for seg in segments])
        
        # 1. Semantic concept identification
        print("   🔍 Identifying crypto concepts...")
        concepts = self.semantic_engine.identify_concepts(full_text)
        
        # 2. Price target extraction
        print("   💰 Extracting price targets...")
        price_targets = self.semantic_engine.extract_price_targets(full_text)
        
        # 3. Argument flow mapping
        print("   🏗️ Mapping argument structure...")
        argument_flow = self.argument_mapper.map_argument_flow(segments)
        
        # 4. Audio conviction analysis (if available)
        audio_conviction = {}
        if audio_file and self.audio_analyzer:
            print("   🎙️ Analyzing voice conviction...")
            # Create prediction objects for audio analysis
            predictions_for_audio = []
            for target in price_targets:
                # Find segment containing this price target
                for seg in segments:
                    if target['raw_text'] in seg['text'].lower():
                        predictions_for_audio.append({
                            'time': seg['time'],
                            'text': seg['text']
                        })
                        break
            
            audio_conviction = self.audio_analyzer.analyze_prediction_conviction(
                audio_file, predictions_for_audio
            )
        
        # 5. Investment signal generation
        print("   📊 Generating investment signals...")
        analysis_data = {
            'concepts': concepts,
            'price_targets': price_targets,
            'argument_flow': argument_flow,
            'audio_conviction': audio_conviction
        }
        
        investment_signals = self.decision_engine.generate_investment_signals(analysis_data)
        
        # 6. Narrative synthesis
        print("   📝 Synthesizing coherent narrative...")
        analysis_data['investment_signals'] = investment_signals
        narrative = self.narrative_synthesizer.synthesize_narrative(analysis_data)
        
        return {
            'concepts': concepts,
            'price_targets': price_targets,
            'argument_flow': argument_flow,
            'audio_conviction': audio_conviction,
            'investment_signals': investment_signals,
            'narrative': narrative,
            'total_segments': len(segments)
        }
    
    def generate_investment_report(self, analysis: Dict, video_id: str) -> str:
        """Generate comprehensive investment intelligence report"""
        
        narrative = analysis['narrative']
        signals = analysis['investment_signals']
        concepts = analysis['concepts']
        audio_conviction = analysis['audio_conviction']
        
        report = f"""
🎯 INVESTMENT INTELLIGENCE REPORT
Video ID: {video_id}
Analysis Engine: TruScraper v7 - Investment Intelligence
{'='*70}

📊 EXECUTIVE SUMMARY
{narrative['executive_summary']}

🎯 MAIN INVESTMENT THESIS
{narrative['main_thesis']}

💼 INVESTMENT CASE
{narrative['investment_case']}

🚨 RISK ASSESSMENT
{narrative['risk_assessment']}

⚡ ACTION PLAN
{narrative['action_plan']}

🔥 INVESTMENT SIGNALS ({len(signals)} generated)
"""
        
        # Group signals by conviction level
        for conviction_level in ConvictionLevel:
            level_signals = [s for s in signals if s.conviction == conviction_level]
            if level_signals:
                emoji_map = {
                    ConvictionLevel.EXTREME: '🚀',
                    ConvictionLevel.HIGH: '🔥', 
                    ConvictionLevel.MEDIUM: '⚠️',
                    ConvictionLevel.LOW: '🤔',
                    ConvictionLevel.UNCERTAIN: '❓'
                }
                
                report += f"\n{emoji_map[conviction_level]} {conviction_level.value.upper()} CONVICTION:\n"
                
                for signal in level_signals:
                    report += f"  • {signal.action} {signal.asset}\n"
                    report += f"    Timeframe: {signal.timeframe}\n"
                    report += f"    Position Size: {signal.position_size}\n"
                    report += f"    Risk Level: {signal.risk_level}\n"
                    report += f"    Rationale: {signal.rationale}\n"
                    
                    if signal.audio_conviction:
                        report += f"    🎙️ Voice Conviction: {signal.audio_conviction:.2f}/1.0\n"
                    report += "\n"
        
        # Audio analysis section
        if audio_conviction and 'conviction_results' in audio_conviction:
            report += f"""
🎙️ VOICE CONVICTION ANALYSIS
• Predictions analyzed: {len(audio_conviction['conviction_results'])}
• Speaker profile: {audio_conviction.get('overall_speaker_profile', {}).get('speaker_type', 'Unknown')}

Top Voice Conviction Moments:
"""
            for i, result in enumerate(audio_conviction['conviction_results'][:3]):
                conv = result['audio_conviction']
                report += f"  {i+1}. Conviction Score: {conv['confidence_score']:.2f}/1.0\n"
                report += f"     Category: {conv['conviction_category']}\n"
                report += f"     Investment Weight: {conv['investment_weight']:.2f}\n\n"
        
        # Semantic concepts
        if concepts:
            report += f"""
🧠 SEMANTIC ANALYSIS
Key Concepts Identified:
"""
            for concept in concepts:
                if concept.confidence > 0.3:
                    report += f"  • {concept.concept.replace('_', ' ').title()}\n"
                    report += f"    Definition: {concept.definition}\n"
                    report += f"    Confidence: {concept.confidence:.2f}\n"
                    report += f"    Implications: {'; '.join(concept.implications[:2])}\n\n"
        
        report += f"""
📈 ANALYSIS METRICS
• Total segments processed: {analysis['total_segments']}
• Concepts identified: {len(concepts)}
• Investment signals: {len(signals)}
• Audio analysis: {'✅ Complete' if audio_conviction else '❌ Not available'}

{'='*70}
🎯 INVESTMENT INTELLIGENCE COMPLETE
This analysis provides actionable investment recommendations based on:
• Semantic understanding of crypto concepts
• Logical argument flow mapping  
• Voice conviction detection
• Professional investment decision framework

⚠️  DISCLAIMER: This is AI analysis for informational purposes only.
Always do your own research and consult financial advisors.
"""
        
        return report

def main():
    """Main execution"""
    if len(sys.argv) != 2:
        print("Usage: python3 truscraper_v7.py <youtube_url>")
        print("Features: Investment Intelligence Engine with voice conviction analysis")
        sys.exit(1)
    
    # Check dependencies
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except:
        print("❌ yt-dlp not installed! Run: pip install yt-dlp")
        sys.exit(1)
    
    scraper = TruScraperV7()
    youtube_url = sys.argv[1]
    
    try:
        print("🚀 Starting Investment Intelligence Analysis...")
        
        # Extract video ID
        video_id = scraper.extract_video_id(youtube_url)
        print(f"📹 Video ID: {video_id}")
        
        # Download and parse subtitles
        segments = scraper.download_and_parse_subtitles(youtube_url, video_id)
        if not segments:
            print("❌ Failed to get subtitles")
            sys.exit(1)
        
        print(f"📝 Processed {len(segments)} transcript segments")
        
        # Download audio for conviction analysis
        audio_file = None
        if AUDIO_AVAILABLE:
            audio_file = scraper.download_audio_for_conviction(youtube_url, video_id)
        
        # Run comprehensive analysis
        analysis = scraper.run_comprehensive_analysis(segments, audio_file)
        
        # Generate investment report
        report = scraper.generate_investment_report(analysis, video_id)
        
        # Save and display
        output_file = f"investment_intelligence_{video_id}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n" + report)
        print(f"\n💾 Investment intelligence report saved: {output_file}")
        
        # Cleanup
        if audio_file:
            try:
                os.remove(audio_file)
                print(f"🗑️ Cleaned up audio file")
            except:
                pass
        
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
