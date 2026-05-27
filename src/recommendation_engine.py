"""
AI-Powered Recommendation Engine for Theft Detection System

Provides intelligent recommendations and action plans based on prediction results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta


class RecommendationEngine:
    """
    Smart recommendation system that provides context-aware actions
    for both theft and honest customer predictions.
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'critical': 0.85,
            'high': 0.70,
            'medium': 0.50,
            'low': 0.30
        }
    
    def generate_theft_recommendations(
        self,
        probability: float,
        features: Dict[str, float],
        customer_id: str
    ) -> Dict:
        """
        Generate comprehensive recommendations for suspected theft cases.
        
        Args:
            probability: Theft probability score
            features: Customer feature dictionary
            customer_id: Customer identifier
        
        Returns:
            Dictionary with risk level, actions, timeline, and resources
        """
        risk_level = self._assess_risk_level(probability)
        risk_score = int(probability * 100)
        
        # Analyze consumption patterns
        pattern_analysis = self._analyze_consumption_pattern(features)
        
        # Generate immediate actions
        immediate_actions = self._get_immediate_actions(risk_level, pattern_analysis)
        
        # Generate follow-up plan
        follow_up_plan = self._create_follow_up_plan(risk_level, pattern_analysis)
        
        # Calculate estimated loss
        estimated_loss = self._estimate_financial_loss(features)
        
        # Generate investigation checklist
        investigation_steps = self._create_investigation_checklist(pattern_analysis)
        
        # Suggest preventive measures
        preventive_measures = self._suggest_preventive_measures(pattern_analysis)
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'confidence': self._calculate_confidence(probability, features),
            'estimated_loss_usd': estimated_loss,
            'pattern_analysis': pattern_analysis,
            'immediate_actions': immediate_actions,
            'investigation_checklist': investigation_steps,
            'follow_up_plan': follow_up_plan,
            'preventive_measures': preventive_measures,
            'priority': self._determine_priority(risk_level, estimated_loss),
            'recommended_timeline': self._get_response_timeline(risk_level)
        }
    
    def generate_honest_recommendations(
        self,
        probability: float,
        features: Dict[str, float],
        customer_id: str
    ) -> Dict:
        """
        Generate engagement strategies for honest customers.
        
        Args:
            probability: Honest probability score (1 - theft_prob)
            features: Customer feature dictionary
            customer_id: Customer identifier
        
        Returns:
            Dictionary with engagement strategies and optimization tips
        """
        confidence_level = int((1 - probability) * 100)  # Honest confidence
        
        # Analyze usage efficiency
        efficiency_analysis = self._analyze_usage_efficiency(features)
        
        # Generate optimization tips
        optimization_tips = self._generate_optimization_tips(features, efficiency_analysis)
        
        # Suggest engagement strategies
        engagement_strategies = self._suggest_engagement_strategies(efficiency_analysis)
        
        # Calculate potential savings
        potential_savings = self._calculate_potential_savings(features, efficiency_analysis)
        
        return {
            'status': 'honest',
            'confidence': confidence_level,
            'usage_profile': self._classify_usage_profile(features),
            'efficiency_score': efficiency_analysis['score'],
            'efficiency_analysis': efficiency_analysis,
            'optimization_tips': optimization_tips,
            'engagement_strategies': engagement_strategies,
            'potential_savings_usd': potential_savings,
            'loyalty_recommendations': self._suggest_loyalty_programs(features)
        }
    
    def _assess_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability."""
        if probability >= self.risk_thresholds['critical']:
            return 'CRITICAL'
        elif probability >= self.risk_thresholds['high']:
            return 'HIGH'
        elif probability >= self.risk_thresholds['medium']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _analyze_consumption_pattern(self, features: Dict[str, float]) -> Dict:
        """Analyze consumption patterns to identify anomalies."""
        patterns = {}
        
        # Check for sudden drops
        sudden_drops = features.get('sudden_drop_count', 0)
        patterns['sudden_drops'] = {
            'count': int(sudden_drops),
            'severity': 'high' if sudden_drops > 3 else ('moderate' if sudden_drops > 1 else 'low'),
            'indicator': 'Critical' if sudden_drops > 3 else 'Notable' if sudden_drops > 0 else 'Normal'
        }
        
        # Check volatility
        volatility = features.get('volatility_index', 0)
        patterns['volatility'] = {
            'value': round(volatility, 3),
            'level': 'high' if volatility > 0.5 else ('moderate' if volatility > 0.3 else 'low'),
            'indicator': 'Erratic' if volatility > 0.5 else 'Variable' if volatility > 0.3 else 'Stable'
        }
        
        # Check for zero consumption days
        zero_days = features.get('zero_day_count', 0)
        patterns['zero_consumption_days'] = {
            'count': int(zero_days),
            'percentage': round((zero_days / 26) * 100, 1),  # Assuming 26 days
            'indicator': 'Suspicious' if zero_days > 10 else 'Moderate' if zero_days > 5 else 'Normal'
        }
        
        # Check trend
        slope = features.get('slope_full', 0)
        patterns['trend'] = {
            'direction': 'declining' if slope < -0.1 else ('stable' if abs(slope) <= 0.1 else 'increasing'),
            'slope': round(slope, 4),
            'indicator': 'Sharp Decline' if slope < -0.3 else 'Declining' if slope < -0.1 else 'Normal'
        }
        
        # Check coefficient of variation
        coef_var = features.get('coef_var', 0)
        patterns['consistency'] = {
            'value': round(coef_var, 3),
            'level': 'inconsistent' if coef_var > 1.0 else ('variable' if coef_var > 0.5 else 'consistent'),
            'indicator': 'Highly Variable' if coef_var > 1.0 else 'Variable' if coef_var > 0.5 else 'Consistent'
        }
        
        return patterns
    
    def _get_immediate_actions(self, risk_level: str, pattern_analysis: Dict) -> List[Dict]:
        """Generate prioritized immediate actions."""
        actions = []
        
        if risk_level in ['CRITICAL', 'HIGH']:
            actions.append({
                'priority': 1,
                'action': 'Physical Meter Inspection',
                'description': 'Dispatch field team for immediate meter inspection and tampering verification',
                'timeline': '24-48 hours',
                'responsible': 'Field Operations Team',
                'icon': 'inspection'
            })
            
            actions.append({
                'priority': 2,
                'action': 'Usage Audit',
                'description': 'Conduct comprehensive audit of historical consumption patterns',
                'timeline': '2-3 days',
                'responsible': 'Analytics Team',
                'icon': 'audit'
            })
            
            actions.append({
                'priority': 3,
                'action': 'Customer Contact',
                'description': 'Initiate formal communication regarding unusual consumption patterns',
                'timeline': '3-5 days',
                'responsible': 'Customer Relations',
                'icon': 'contact'
            })
        
        if pattern_analysis['sudden_drops']['severity'] == 'high':
            actions.append({
                'priority': 2 if risk_level == 'CRITICAL' else 3,
                'action': 'Bypass Investigation',
                'description': 'Check for unauthorized connections or meter bypass indicators',
                'timeline': '24-48 hours',
                'responsible': 'Security Team',
                'icon': 'security'
            })
        
        if pattern_analysis['zero_consumption_days']['count'] > 10:
            actions.append({
                'priority': 3,
                'action': 'Meter Functionality Check',
                'description': 'Verify meter is functioning correctly and recording consumption',
                'timeline': '48-72 hours',
                'responsible': 'Technical Support',
                'icon': 'technical'
            })
        
        if risk_level in ['MEDIUM', 'LOW']:
            actions.append({
                'priority': 1,
                'action': 'Enhanced Monitoring',
                'description': 'Place customer on watchlist for 30-day enhanced monitoring',
                'timeline': 'Immediate',
                'responsible': 'Monitoring System',
                'icon': 'monitor'
            })
            
            actions.append({
                'priority': 2,
                'action': 'Pattern Analysis',
                'description': 'Deep-dive analysis of consumption patterns vs. similar customers',
                'timeline': '1 week',
                'responsible': 'Data Science Team',
                'icon': 'analytics'
            })
        
        return sorted(actions, key=lambda x: x['priority'])
    
    def _create_investigation_checklist(self, pattern_analysis: Dict) -> List[Dict]:
        """Create investigation checklist based on pattern analysis."""
        checklist = [
            {
                'item': 'Meter Physical Condition',
                'checks': [
                    'Inspect meter for physical tampering or damage',
                    'Check seals and security features',
                    'Verify meter serial number matches records',
                    'Look for unauthorized modifications'
                ],
                'critical': True
            },
            {
                'item': 'Connection Verification',
                'checks': [
                    'Inspect service connection point',
                    'Check for unauthorized connections',
                    'Verify proper grounding',
                    'Look for bypass indicators'
                ],
                'critical': True
            },
            {
                'item': 'Historical Data Review',
                'checks': [
                    'Compare current vs. historical consumption',
                    'Analyze billing discrepancies',
                    'Review previous complaints or issues',
                    'Check for unusual payment patterns'
                ],
                'critical': False
            },
            {
                'item': 'Customer Profile Analysis',
                'checks': [
                    'Review customer information and history',
                    'Check property type and expected usage',
                    'Verify number of occupants',
                    'Compare with similar customer profiles'
                ],
                'critical': False
            }
        ]
        
        if pattern_analysis['sudden_drops']['severity'] == 'high':
            checklist.insert(1, {
                'item': 'Sudden Drop Investigation',
                'checks': [
                    'Identify exact dates of consumption drops',
                    'Correlate with service interruptions or maintenance',
                    'Check for customer relocation or property changes',
                    'Investigate potential meter malfunction'
                ],
                'critical': True
            })
        
        return checklist
    
    def _create_follow_up_plan(self, risk_level: str, pattern_analysis: Dict) -> Dict:
        """Create structured follow-up plan with timeline."""
        if risk_level == 'CRITICAL':
            return {
                'day_1_3': [
                    'Complete immediate physical inspection',
                    'Document all findings with photos',
                    'Initiate formal investigation process'
                ],
                'day_4_7': [
                    'Analyze investigation results',
                    'Prepare case documentation',
                    'Contact customer with findings',
                    'Determine corrective actions needed'
                ],
                'day_8_14': [
                    'Implement corrective measures',
                    'Install enhanced monitoring if needed',
                    'Complete billing adjustments',
                    'Schedule follow-up inspection'
                ],
                'day_15_30': [
                    'Monitor post-intervention consumption',
                    'Verify pattern normalization',
                    'Close case or escalate as needed',
                    'Update customer profile'
                ]
            }
        elif risk_level == 'HIGH':
            return {
                'week_1': [
                    'Schedule and complete meter inspection',
                    'Analyze historical patterns',
                    'Prepare preliminary findings'
                ],
                'week_2': [
                    'Customer outreach and communication',
                    'Additional investigation if needed',
                    'Document all interactions'
                ],
                'week_3_4': [
                    'Implement monitoring enhancements',
                    'Verify consumption patterns',
                    'Complete case documentation'
                ]
            }
        else:
            return {
                'week_1_2': [
                    'Place on enhanced monitoring watchlist',
                    'Automated pattern analysis'
                ],
                'week_3_4': [
                    'Review monitoring results',
                    'Determine if further action needed'
                ]
            }
    
    def _suggest_preventive_measures(self, pattern_analysis: Dict) -> List[Dict]:
        """Suggest preventive measures to avoid future theft."""
        measures = [
            {
                'measure': 'Smart Meter Upgrade',
                'description': 'Install tamper-resistant smart meters with remote monitoring',
                'benefit': 'Real-time alerts, remote disconnection, detailed usage data',
                'cost': 'Medium',
                'effectiveness': 'Very High'
            },
            {
                'measure': 'Regular Audits',
                'description': 'Implement scheduled consumption audits for high-risk customers',
                'benefit': 'Early detection, deterrent effect',
                'cost': 'Low',
                'effectiveness': 'High'
            },
            {
                'measure': 'Customer Education',
                'description': 'Awareness campaigns about theft consequences and reporting',
                'benefit': 'Reduced incidents, community vigilance',
                'cost': 'Low',
                'effectiveness': 'Medium'
            },
            {
                'measure': 'Enhanced Security',
                'description': 'Install tamper-evident seals and physical security measures',
                'benefit': 'Physical tampering prevention',
                'cost': 'Low',
                'effectiveness': 'Medium'
            }
        ]
        
        if pattern_analysis['sudden_drops']['severity'] == 'high':
            measures.insert(1, {
                'measure': 'Consumption Pattern Alerts',
                'description': 'Automated alerts for sudden consumption changes',
                'benefit': 'Immediate detection of anomalies',
                'cost': 'Low',
                'effectiveness': 'High'
            })
        
        return measures
    
    def _estimate_financial_loss(self, features: Dict[str, float]) -> float:
        """Estimate potential financial loss from theft."""
        mean_consumption = features.get('mean', 100)
        sudden_drops = features.get('sudden_drop_count', 0)
        zero_days = features.get('zero_day_count', 0)
        
        # Rough estimation: average consumption * affected days * rate
        rate_per_unit = 0.12  # $0.12 per kWh (example)
        affected_days = min(zero_days + (sudden_drops * 3), 26)
        
        estimated_loss = mean_consumption * affected_days * rate_per_unit
        return round(estimated_loss, 2)
    
    def _calculate_confidence(self, probability: float, features: Dict) -> str:
        """Calculate model confidence level."""
        if probability > 0.85:
            return 'Very High'
        elif probability > 0.70:
            return 'High'
        elif probability > 0.55:
            return 'Moderate'
        else:
            return 'Low'
    
    def _determine_priority(self, risk_level: str, estimated_loss: float) -> str:
        """Determine case priority."""
        if risk_level == 'CRITICAL' or estimated_loss > 500:
            return 'P1 - Immediate'
        elif risk_level == 'HIGH' or estimated_loss > 200:
            return 'P2 - Urgent'
        elif risk_level == 'MEDIUM':
            return 'P3 - High'
        else:
            return 'P4 - Normal'
    
    def _get_response_timeline(self, risk_level: str) -> str:
        """Get recommended response timeline."""
        timelines = {
            'CRITICAL': '24 hours',
            'HIGH': '2-3 days',
            'MEDIUM': '1 week',
            'LOW': '2 weeks'
        }
        return timelines.get(risk_level, '2 weeks')
    
    def _analyze_usage_efficiency(self, features: Dict[str, float]) -> Dict:
        """Analyze usage efficiency for honest customers."""
        mean = features.get('mean', 0)
        std = features.get('std', 0)
        coef_var = features.get('coef_var', 0)
        zero_days = features.get('zero_day_count', 0)
        
        # Calculate efficiency score (0-100)
        consistency_score = max(0, 100 - (coef_var * 50))
        utilization_score = max(0, 100 - ((zero_days / 26) * 100))
        efficiency_score = int((consistency_score + utilization_score) / 2)
        
        return {
            'score': efficiency_score,
            'grade': 'A' if efficiency_score >= 85 else ('B' if efficiency_score >= 70 else ('C' if efficiency_score >= 55 else 'D')),
            'consistency': {
                'score': int(consistency_score),
                'status': 'Excellent' if consistency_score >= 80 else ('Good' if consistency_score >= 60 else 'Fair')
            },
            'utilization': {
                'score': int(utilization_score),
                'status': 'Optimal' if utilization_score >= 80 else ('Good' if utilization_score >= 60 else 'Needs Improvement')
            },
            'average_daily_usage': round(mean, 2),
            'variability': round(std, 2)
        }
    
    def _generate_optimization_tips(self, features: Dict, efficiency: Dict) -> List[Dict]:
        """Generate personalized optimization tips."""
        tips = []
        
        coef_var = features.get('coef_var', 0)
        if coef_var > 0.5:
            tips.append({
                'category': 'Usage Consistency',
                'tip': 'Stabilize Daily Consumption',
                'description': 'Your usage varies significantly day-to-day. Consider establishing routine usage patterns.',
                'potential_savings': '10-15%',
                'difficulty': 'Easy',
                'icon': 'consistency'
            })
        
        peak_ratio = features.get('peak_day_ratio', 0)
        if peak_ratio > 2.0:
            tips.append({
                'category': 'Peak Usage',
                'tip': 'Reduce Peak Consumption',
                'description': 'Your peak days use significantly more energy. Spread high-consumption activities.',
                'potential_savings': '15-20%',
                'difficulty': 'Medium',
                'icon': 'peak'
            })
        
        zero_days = features.get('zero_day_count', 0)
        if zero_days < 2:
            tips.append({
                'category': 'Energy Management',
                'tip': 'Implement Energy-Free Days',
                'description': 'Consider scheduling occasional low/no-consumption days to reduce overall usage.',
                'potential_savings': '5-10%',
                'difficulty': 'Easy',
                'icon': 'calendar'
            })
        
        tips.append({
            'category': 'Smart Monitoring',
            'tip': 'Enable Real-Time Monitoring',
            'description': 'Set up daily usage alerts to track consumption and identify inefficiencies.',
            'potential_savings': '10-15%',
            'difficulty': 'Easy',
            'icon': 'monitor'
        })
        
        return tips
    
    def _suggest_engagement_strategies(self, efficiency: Dict) -> List[Dict]:
        """Suggest customer engagement strategies."""
        strategies = []
        
        if efficiency['score'] >= 80:
            strategies.append({
                'strategy': 'VIP Recognition Program',
                'description': 'Enroll in VIP program for exemplary customers',
                'benefits': ['Priority support', 'Exclusive rates', 'Early access to new features'],
                'action': 'Auto-enroll eligible'
            })
        
        strategies.append({
            'strategy': 'Usage Insights Dashboard',
            'description': 'Provide personalized consumption analytics',
            'benefits': ['Detailed breakdown', 'Comparison with similar homes', 'Savings recommendations'],
            'action': 'Enable dashboard access'
        })
        
        strategies.append({
            'strategy': 'Efficiency Rewards',
            'description': 'Reward consistent, efficient usage patterns',
            'benefits': ['Cashback opportunities', 'Bill credits', 'Green energy discounts'],
            'action': 'Enroll in rewards program'
        })
        
        return strategies
    
    def _calculate_potential_savings(self, features: Dict, efficiency: Dict) -> float:
        """Calculate potential savings through optimization."""
        mean_consumption = features.get('mean', 100)
        
        # Estimate 10-20% savings potential based on efficiency
        savings_potential = 0.10 if efficiency['score'] >= 70 else 0.15
        rate = 0.12  # Example rate
        
        monthly_savings = mean_consumption * 30 * rate * savings_potential
        return round(monthly_savings, 2)
    
    def _suggest_loyalty_programs(self, features: Dict) -> List[Dict]:
        """Suggest appropriate loyalty programs."""
        return [
            {
                'program': 'Green Energy Initiative',
                'description': 'Opt-in to renewable energy sources',
                'benefit': '5% discount on monthly bills',
                'eligibility': 'All honest customers'
            },
            {
                'program': 'Referral Rewards',
                'description': 'Refer friends and family',
                'benefit': '$25 credit per successful referral',
                'eligibility': 'Active customers'
            },
            {
                'program': 'Auto-Pay Discount',
                'description': 'Enable automatic payments',
                'benefit': '2% monthly discount',
                'eligibility': 'All customers'
            }
        ]
    
    def _classify_usage_profile(self, features: Dict) -> str:
        """Classify customer usage profile."""
        mean = features.get('mean', 0)
        coef_var = features.get('coef_var', 0)
        
        if mean > 150:
            if coef_var < 0.3:
                return 'High-Consistent User'
            else:
                return 'High-Variable User'
        elif mean > 75:
            if coef_var < 0.3:
                return 'Moderate-Consistent User'
            else:
                return 'Moderate-Variable User'
        else:
            if coef_var < 0.3:
                return 'Low-Consistent User'
            else:
                return 'Low-Variable User'
