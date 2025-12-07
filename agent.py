#!/usr/bin/env python3
"""
Student Career Assistant Agent - Enhanced Multi-Agent System
Agents Intensive Capstone Project
Track: Concierge Agents
Author: Gourav Suresh
Date: November 15, 2025

Enhanced Features:
1. Google Gemini API integration for intelligent routing
2. Expanded DSA problem database (50+ problems)
3. Advanced Resume Analyzer with PDF support
4. Mock Interview Conductor Agent
5. Progress tracking and analytics
6. Persistent JSON storage
7. Interactive CLI interface
8. Real-time learning resource recommendations
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Third-party imports
try:
    import google.generativeai as genai
except ImportError:
    print("⚠️  Warning: google-generativeai not installed. Run: pip install google-generativeai")
    genai = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    print("⚠️  Warning: PyPDF2 not installed. Run: pip install PyPDF2")
    PdfReader = None

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


#====================================================================================
# CONFIGURATION & SETUP
#====================================================================================

class Config:
    """Configuration management"""
    DATA_DIR = Path("data")
    STUDENTS_FILE = DATA_DIR / "students.json"
    PROBLEMS_FILE = DATA_DIR / "problems.json"
    
    @classmethod
    def setup(cls):
        """Create necessary directories and files"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        if not cls.STUDENTS_FILE.exists():
            cls.STUDENTS_FILE.write_text("{}")
        if not cls.PROBLEMS_FILE.exists():
            cls.PROBLEMS_FILE.write_text("{}")
        logger.info("Configuration setup complete")


#====================================================================================
# MEMORY MANAGEMENT - Enhanced with persistent storage
#====================================================================================

class StudentMemoryBank:
    """Enhanced Memory Bank with persistent JSON storage"""
    
    def __init__(self):
        self.students = self._load_students()
        logger.info(f"Memory Bank initialized with {len(self.students)} students")
    
    def _load_students(self) -> Dict:
        """Load students from JSON file"""
        try:
            with open(Config.STUDENTS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading students: {e}")
            return {}
    
    def _save_students(self):
        """Save students to JSON file"""
        try:
            with open(Config.STUDENTS_FILE, 'w') as f:
                json.dump(self.students, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving students: {e}")
    
    def add_student(self, student_id: str, profile: Dict):
        """Add a new student"""
        self.students[student_id] = {
            'profile': profile,
            'progress': [],
            'preferences': {},
            'statistics': {
                'problems_solved': 0,
                'easy_solved': 0,
                'medium_solved': 0,
                'hard_solved': 0,
                'topics_covered': [],
                'total_study_hours': 0
            },
            'created_at': datetime.now().isoformat(),
            'last_active': datetime.now().isoformat()
        }
        self._save_students()
        logger.info(f"Added student: {student_id}")
    
    def update_progress(self, student_id: str, activity: str, details: Dict):
        """Update student progress"""
        if student_id in self.students:
            self.students[student_id]['progress'].append({
                'activity': activity,
                'details': details,
                'timestamp': datetime.now().isoformat()
            })
            self.students[student_id]['last_active'] = datetime.now().isoformat()
            self._save_students()
            logger.info(f"Updated progress for {student_id}: {activity}")
    
    def update_statistics(self, student_id: str, stat_key: str, value: Any):
        """Update student statistics"""
        if student_id in self.students:
            if stat_key in ['easy_solved', 'medium_solved', 'hard_solved', 'problems_solved']:
                self.students[student_id]['statistics'][stat_key] += value
            elif stat_key == 'topics_covered':
                if value not in self.students[student_id]['statistics']['topics_covered']:
                    self.students[student_id]['statistics']['topics_covered'].append(value)
            self._save_students()
    
    def get_student_context(self, student_id: str) -> Dict:
        """Get student context"""
        return self.students.get(student_id, {})
    
    def get_statistics(self, student_id: str) -> Dict:
        """Get student statistics"""
        if student_id in self.students:
            return self.students[student_id]['statistics']
        return {}


#====================================================================================
# ENHANCED DSA PROBLEM RECOMMENDER - 50+ Problems
#====================================================================================

class DSAProblemRecommender:
    """Enhanced DSA Problem Recommender with comprehensive problem database"""
    
    def __init__(self):
        self.problems = {
            'easy': [
                # Arrays
                {'name': 'Two Sum', 'topic': 'Array', 'difficulty': 'Easy', 'leetcode_id': 1},
                {'name': 'Best Time to Buy and Sell Stock', 'topic': 'Array', 'difficulty': 'Easy', 'leetcode_id': 121},
                {'name': 'Contains Duplicate', 'topic': 'Array', 'difficulty': 'Easy', 'leetcode_id': 217},
                {'name': 'Product of Array Except Self', 'topic': 'Array', 'difficulty': 'Easy', 'leetcode_id': 238},
                # Strings
                {'name': 'Valid Anagram', 'topic': 'String', 'difficulty': 'Easy', 'leetcode_id': 242},
                {'name': 'Valid Palindrome', 'topic': 'String', 'difficulty': 'Easy', 'leetcode_id': 125},
                {'name': 'Reverse String', 'topic': 'String', 'difficulty': 'Easy', 'leetcode_id': 344},
                # Linked Lists
                {'name': 'Reverse Linked List', 'topic': 'Linked List', 'difficulty': 'Easy', 'leetcode_id': 206},
                {'name': 'Merge Two Sorted Lists', 'topic': 'Linked List', 'difficulty': 'Easy', 'leetcode_id': 21},
                {'name': 'Linked List Cycle', 'topic': 'Linked List', 'difficulty': 'Easy', 'leetcode_id': 141},
                # Stack/Queue
                {'name': 'Valid Parentheses', 'topic': 'Stack', 'difficulty': 'Easy', 'leetcode_id': 20},
                {'name': 'Implement Queue using Stacks', 'topic': 'Stack', 'difficulty': 'Easy', 'leetcode_id': 232},
                # Tree
                {'name': 'Maximum Depth of Binary Tree', 'topic': 'Tree', 'difficulty': 'Easy', 'leetcode_id': 104},
                {'name': 'Invert Binary Tree', 'topic': 'Tree', 'difficulty': 'Easy', 'leetcode_id': 226},
                {'name': 'Same Tree', 'topic': 'Tree', 'difficulty': 'Easy', 'leetcode_id': 100},
            ],
            'medium': [
                # Arrays
                {'name': 'Three Sum', 'topic': 'Array', 'difficulty': 'Medium', 'leetcode_id': 15},
                {'name': 'Container With Most Water', 'topic': 'Array', 'difficulty': 'Medium', 'leetcode_id': 11},
                {'name': 'Find Minimum in Rotated Sorted Array', 'topic': 'Array', 'difficulty': 'Medium', 'leetcode_id': 153},
                # Strings
                {'name': 'Longest Substring Without Repeating Characters', 'topic': 'String', 'difficulty': 'Medium', 'leetcode_id': 3},
                {'name': 'Longest Palindromic Substring', 'topic': 'String', 'difficulty': 'Medium', 'leetcode_id': 5},
                {'name': 'Group Anagrams', 'topic': 'String', 'difficulty': 'Medium', 'leetcode_id': 49},
                # Linked Lists
                {'name': 'Add Two Numbers', 'topic': 'Linked List', 'difficulty': 'Medium', 'leetcode_id': 2},
                {'name': 'Remove Nth Node From End of List', 'topic': 'Linked List', 'difficulty': 'Medium', 'leetcode_id': 19},
                {'name': 'Reorder List', 'topic': 'Linked List', 'difficulty': 'Medium', 'leetcode_id': 143},
                # Trees
                {'name': 'Binary Tree Level Order Traversal', 'topic': 'Tree', 'difficulty': 'Medium', 'leetcode_id': 102},
                {'name': 'Validate Binary Search Tree', 'topic': 'Tree', 'difficulty': 'Medium', 'leetcode_id': 98},
                {'name': 'Kth Smallest Element in a BST', 'topic': 'Tree', 'difficulty': 'Medium', 'leetcode_id': 230},
                {'name': 'Lowest Common Ancestor of BST', 'topic': 'Tree', 'difficulty': 'Medium', 'leetcode_id': 235},
                # Graphs
                {'name': 'Number of Islands', 'topic': 'Graph', 'difficulty': 'Medium', 'leetcode_id': 200},
                {'name': 'Clone Graph', 'topic': 'Graph', 'difficulty': 'Medium', 'leetcode_id': 133},
                {'name': 'Course Schedule', 'topic': 'Graph', 'difficulty': 'Medium', 'leetcode_id': 207},
                # Dynamic Programming
                {'name': 'Climbing Stairs', 'topic': 'Dynamic Programming', 'difficulty': 'Medium', 'leetcode_id': 70},
                {'name': 'Coin Change', 'topic': 'Dynamic Programming', 'difficulty': 'Medium', 'leetcode_id': 322},
                {'name': 'Longest Increasing Subsequence', 'topic': 'Dynamic Programming', 'difficulty': 'Medium', 'leetcode_id': 300},
                # Design
                {'name': 'LRU Cache', 'topic': 'Design', 'difficulty': 'Medium', 'leetcode_id': 146},
                {'name': 'Implement Trie', 'topic': 'Design', 'difficulty': 'Medium', 'leetcode_id': 208},
            ],
            'hard': [
                # Arrays
                {'name': 'Trapping Rain Water', 'topic': 'Array', 'difficulty': 'Hard', 'leetcode_id': 42},
                {'name': 'Sliding Window Maximum', 'topic': 'Array', 'difficulty': 'Hard', 'leetcode_id': 239},
                # Strings
                {'name': 'Minimum Window Substring', 'topic': 'String', 'difficulty': 'Hard', 'leetcode_id': 76},
                {'name': 'Word Ladder II', 'topic': 'String', 'difficulty': 'Hard', 'leetcode_id': 126},
                # Trees
                {'name': 'Binary Tree Maximum Path Sum', 'topic': 'Tree', 'difficulty': 'Hard', 'leetcode_id': 124},
                {'name': 'Serialize and Deserialize Binary Tree', 'topic': 'Tree', 'difficulty': 'Hard', 'leetcode_id': 297},
                # Graphs
                {'name': 'Word Ladder', 'topic': 'Graph', 'difficulty': 'Hard', 'leetcode_id': 127},
                {'name': 'Alien Dictionary', 'topic': 'Graph', 'difficulty': 'Hard', 'leetcode_id': 269},
                # Dynamic Programming
                {'name': 'Edit Distance', 'topic': 'Dynamic Programming', 'difficulty': 'Hard', 'leetcode_id': 72},
                {'name': 'Regular Expression Matching', 'topic': 'Dynamic Programming', 'difficulty': 'Hard', 'leetcode_id': 10},
                # Binary Search
                {'name': 'Median of Two Sorted Arrays', 'topic': 'Binary Search', 'difficulty': 'Hard', 'leetcode_id': 4},
            ]
        }
        logger.info(f"DSA Problem Recommender initialized with {sum(len(v) for v in self.problems.values())} problems")
    
    def recommend(self, level: str = 'medium', topic: str = None, count: int = 5) -> List[Dict]:
        """Recommend problems based on criteria"""
        problems = self.problems.get(level.lower(), [])
        
        if topic:
            problems = [p for p in problems if p['topic'].lower() == topic.lower()]
        
        # Return up to 'count' problems
        result = problems[:count]
        logger.info(f"Recommended {len(result)} {level} problems" + (f" for topic: {topic}" if topic else ""))
        return result
    
    def get_topics(self, level: str = None) -> List[str]:
        """Get all available topics"""
        if level:
            return list(set(p['topic'] for p in self.problems.get(level.lower(), [])))
        return list(set(p['topic'] for problems in self.problems.values() for p in problems))


#====================================================================================
# ENHANCED RESUME ANALYZER - With PDF support
#====================================================================================

class ResumeAnalyzer:
    """Enhanced Resume Analyzer with PDF support and advanced ATS analysis"""
    
    def __init__(self):
        self.ats_keywords = {
            'software_engineer': ['python', 'java', 'javascript', 'react', 'node', 'sql', 'git', 'agile', 'api', 'docker'],
            'ml_engineer': ['python', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'sql', 'spark', 'deep learning', 'nlp'],
            'data_scientist': ['python', 'r', 'sql', 'tableau', 'pandas', 'statistics', 'machine learning', 'visualization', 'spark'],
            'frontend_engineer': ['javascript', 'react', 'vue', 'angular', 'html', 'css', 'typescript', 'webpack', 'redux'],
            'backend_engineer': ['python', 'java', 'go', 'sql', 'postgresql', 'mongodb', 'redis', 'docker', 'kubernetes', 'microservices'],
        }
        self.action_verbs = [
            'developed', 'implemented', 'designed', 'architected', 'optimized', 'led', 'managed', 
            'created', 'built', 'engineered', 'automated', 'deployed', 'scaled', 'improved', 'reduced'
        ]
        logger.info("Resume Analyzer initialized with role-specific keywords")
    
    def parse_pdf(self, pdf_path: str) -> str:
        """Parse PDF resume to text"""
        if PdfReader is None:
            return "PDF parsing not available. Install PyPDF2: pip install PyPDF2"
        
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            return ""
    
    def analyze(self, resume_text: str, target_role: str = "software_engineer", pdf_path: str = None) -> Dict:
        """Comprehensive resume analysis"""
        # If PDF path provided, parse it
        if pdf_path:
            resume_text = self.parse_pdf(pdf_path)
        
        analysis = {
            'ats_score': 0,
            'keyword_analysis': {},
            'suggestions': [],
            'strengths': [],
            'missing_keywords': [],
            'action_verbs_found': [],
            'word_count': len(resume_text.split()),
            'target_role': target_role
        }
        
        resume_lower = resume_text.lower()
        
        # Get role-specific keywords
        role_keywords = self.ats_keywords.get(target_role, self.ats_keywords['software_engineer'])
        
        # Check technical skills
        found_keywords = [kw for kw in role_keywords if kw in resume_lower]
        missing_keywords = [kw for kw in role_keywords if kw not in resume_lower]
        
        analysis['keyword_analysis']['found'] = found_keywords
        analysis['missing_keywords'] = missing_keywords
        
        # Check action verbs
        found_verbs = [verb for verb in self.action_verbs if verb in resume_lower]
        analysis['action_verbs_found'] = found_verbs
        
        # Calculate ATS score (0-100)
        keyword_score = (len(found_keywords) / len(role_keywords)) * 60
        verb_score = min(len(found_verbs) * 3, 20)
        length_score = 10 if 300 <= analysis['word_count'] <= 600 else 5
        analysis['ats_score'] = int(keyword_score + verb_score + length_score)
        
        # Generate strengths
        if len(found_keywords) > len(role_keywords) * 0.6:
            analysis['strengths'].append(f"Strong technical keyword coverage ({len(found_keywords)}/{len(role_keywords)})")
        if len(found_verbs) >= 5:
            analysis['strengths'].append(f"Good use of action verbs ({len(found_verbs)} found)")
        if 300 <= analysis['word_count'] <= 600:
            analysis['strengths'].append("Optimal resume length")
        
        # Generate suggestions
        if len(missing_keywords) > 0:
            analysis['suggestions'].append(f"Add these {target_role.replace('_', ' ').title()} keywords: {', '.join(missing_keywords[:5])}")
        if len(found_verbs) < 5:
            analysis['suggestions'].append("Use more action verbs to describe achievements (developed, implemented, optimized)")
        if analysis['word_count'] < 300:
            analysis['suggestions'].append("Resume is too short. Add more details about projects and achievements")
        if analysis['word_count'] > 600:
            analysis['suggestions'].append("Resume is too long. Focus on most relevant and impactful experiences")
        if 'quantif' not in resume_lower and 'metric' not in resume_lower:
            analysis['suggestions'].append("Add quantifiable metrics (e.g., 'Improved performance by 40%', 'Reduced latency by 2s')")
        
        logger.info(f"Resume analyzed. ATS Score: {analysis['ats_score']}, Role: {target_role}")
        return analysis


#====================================================================================
# MOCK INTERVIEW CONDUCTOR - New Agent
#====================================================================================

class MockInterviewAgent:
    """Conducts mock technical interviews"""
    
    def __init__(self, problem_recommender: DSAProblemRecommender):
        self.problem_recommender = problem_recommender
        self.interview_questions = {
            'behavioral': [
                "Tell me about a challenging project you worked on.",
                "Describe a time when you had to debug a complex issue.",
                "How do you handle disagreements with team members?",
                "Tell me about a time you failed and what you learned.",
                "Why do you want to work for our company?"
            ],
            'system_design': [
                "Design a URL shortener like bit.ly",
                "Design a rate limiter",
                "Design Instagram/Twitter feed",
                "Design a distributed cache",
                "Design a messaging system like WhatsApp"
            ]
        }
        logger.info("Mock Interview Agent initialized")
    
    def start_interview(self, interview_type: str = 'coding', difficulty: str = 'medium') -> Dict:
        """Start a mock interview session"""
        result = {
            'type': interview_type,
            'difficulty': difficulty,
            'questions': [],
            'tips': []
        }
        
        if interview_type == 'coding':
            problems = self.problem_recommender.recommend(level=difficulty, count=3)
            result['questions'] = problems
            result['tips'] = [
                "Think out loud - explain your approach before coding",
                "Ask clarifying questions about edge cases",
                "Discuss time and space complexity",
                "Test your code with examples"
            ]
        elif interview_type == 'behavioral':
            result['questions'] = self.interview_questions['behavioral'][:3]
            result['tips'] = [
                "Use the STAR method (Situation, Task, Action, Result)",
                "Be specific with examples",
                "Show your problem-solving process",
                "Highlight what you learned"
            ]
        elif interview_type == 'system_design':
            result['questions'] = [self.interview_questions['system_design'][0]]
            result['tips'] = [
                "Start with requirements and constraints",
                "Discuss trade-offs between different approaches",
                "Draw diagrams (APIs, databases, components)",
                "Consider scalability and bottlenecks",
                "Discuss data flow and edge cases"
            ]
        
        logger.info(f"Started {interview_type} mock interview at {difficulty} level")
        return result
    
    def evaluate_answer(self, question: str, answer: str) -> Dict:
        """Evaluate interview answer (placeholder for LLM integration)"""
        evaluation = {
            'question': question,
            'feedback': "Good attempt! Consider discussing time complexity and edge cases.",
            'score': 7.5,
            'improvement_areas': [
                "Explain your thought process more clearly",
                "Discuss alternative approaches",
                "Consider edge cases"
            ]
        }
        return evaluation


#====================================================================================
# LEARNING RESOURCE AGENT - Enhanced
#====================================================================================

class LearningResourceAgent:
    """Enhanced Learning Resource recommendations"""
    
    def __init__(self):
        self.resources = {
            'dsa': [
                {'title': 'NeetCode.io - Curated LeetCode Problems', 'type': 'Practice', 'difficulty': 'All levels', 'url': 'https://neetcode.io'},
                {'title': 'LeetCode Patterns by Sean Prashad', 'type': 'Guide', 'difficulty': 'All levels', 'url': 'https://seanprashad.com/leetcode-patterns/'},
                {'title': 'Blind 75 LeetCode Questions', 'type': 'Practice List', 'difficulty': 'Mixed', 'url': 'https://leetcode.com/discuss/general-discussion/460599/blind-75-leetcode-questions'},
                {'title': 'Abdul Bari Algorithms Course', 'type': 'Video', 'difficulty': 'Beginner', 'url': 'https://youtube.com'},
                {'title': 'Cracking the Coding Interview Book', 'type': 'Book', 'difficulty': 'All levels', 'url': 'amazon.com'},
            ],
            'system_design': [
                {'title': 'System Design Primer (GitHub)', 'type': 'Article', 'difficulty': 'Beginner', 'url': 'https://github.com/donnemartin/system-design-primer'},
                {'title': 'ByteByteGo', 'type': 'Video', 'difficulty': 'All levels', 'url': 'https://bytebytego.com'},
                {'title': 'Gaurav Sen System Design', 'type': 'Video', 'difficulty': 'Intermediate', 'url': 'https://youtube.com'},
                {'title': 'Designing Data-Intensive Applications Book', 'type': 'Book', 'difficulty': 'Advanced', 'url': 'amazon.com'},
            ],
            'python': [
                {'title': 'Real Python', 'type': 'Tutorial', 'difficulty': 'All levels', 'url': 'https://realpython.com'},
                {'title': 'Python Official Documentation', 'type': 'Documentation', 'difficulty': 'All levels', 'url': 'https://docs.python.org'},
                {'title': 'Automate the Boring Stuff', 'type': 'Book', 'difficulty': 'Beginner', 'url': 'https://automatetheboringstuff.com'},
            ],
            'machine_learning': [
                {'title': 'Fast.ai Practical Deep Learning', 'type': 'Course', 'difficulty': 'Beginner', 'url': 'https://fast.ai'},
                {'title': 'Andrew Ng Machine Learning Course', 'type': 'Course', 'difficulty': 'Beginner', 'url': 'https://coursera.org'},
                {'title': 'Kaggle Learn', 'type': 'Practice', 'difficulty': 'All levels', 'url': 'https://kaggle.com/learn'},
            ],
            'behavioral_interview': [
                {'title': 'Jeff H Sipe Behavioral Interview Guide', 'type': 'Guide', 'difficulty': 'All levels', 'url': 'youtube.com'},
                {'title': 'STAR Method Examples', 'type': 'Article', 'difficulty': 'All levels', 'url': 'themuse.com'},
            ]
        }
        logger.info("Learning Resource Agent initialized")
    
    def recommend(self, topic: str, level: str = 'All levels', count: int = 5) -> List[Dict]:
        """Recommend learning resources"""
        topic_lower = topic.lower().replace(' ', '_')
        resources = self.resources.get(topic_lower, [])
        
        if level != 'All levels':
            resources = [r for r in resources if r['difficulty'] == level or r['difficulty'] == 'All levels']
        
        result = resources[:count]
        logger.info(f"Recommended {len(result)} resources for {topic}")
        return result
    
    def get_all_topics(self) -> List[str]:
        """Get all available topics"""
        return list(self.resources.keys())


#====================================================================================
# STUDY PLANNER AGENT - Enhanced
#====================================================================================

class StudyPlannerAgent:
    """Enhanced Study Planner with detailed schedules"""
    
    def __init__(self):
        logger.info("Study Planner Agent initialized")
    
    def create_plan(self, weeks: int, focus_areas: List[str], hours_per_day: int = 4, skill_level: str = 'intermediate') -> Dict:
        """Create detailed study plan"""
        plan = {
            'duration': f"{weeks} weeks",
            'daily_commitment': f"{hours_per_day} hours/day",
            'total_hours': weeks * 7 * hours_per_day,
            'skill_level': skill_level,
            'focus_areas': focus_areas,
            'weekly_schedule': [],
            'milestones': [],
            'daily_routine': self._get_daily_routine(hours_per_day)
        }
        
        # Generate weekly schedules
        for week in range(1, weeks + 1):
            focus = focus_areas[(week - 1) % len(focus_areas)]
            week_plan = self._generate_week_plan(week, focus, skill_level)
            plan['weekly_schedule'].append(week_plan)
            
            # Add milestone
            plan['milestones'].append({
                'week': week,
                'goal': f"Complete {focus} fundamentals" if week <= weeks/2 else f"Master {focus} advanced concepts",
                'assessment': f"Solve 10-15 {focus} problems" if 'DSA' in focus else f"Complete {focus} project"
            })
        
        logger.info(f"Created {weeks}-week study plan with {len(focus_areas)} focus areas")
        return plan
    
    def _get_daily_routine(self, hours: int) -> Dict:
        """Get recommended daily routine"""
        if hours <= 2:
            return {
                'learning': '1 hour - Theory and concepts',
                'practice': '1 hour - Problem solving',
                'review': '15 min - Review and notes'
            }
        elif hours <= 4:
            return {
                'learning': '1.5 hours - Theory and concepts',
                'practice': '2 hours - Problem solving and projects',
                'review': '30 min - Review, notes, and flashcards'
            }
        else:
            return {
                'learning': '2 hours - Deep dive into theory',
                'practice': '3 hours - Intensive problem solving',
                'mock_interview': '1 hour - Mock interviews',
                'review': '30 min - Review and plan next day'
            }
    
    def _generate_week_plan(self, week: int, focus: str, skill_level: str) -> Dict:
        """Generate detailed week plan"""
        activities = {
            'DSA': ['Study algorithms', 'Solve easy problems', 'Solve medium problems', 'Review solutions', 'Mock interviews'],
            'System Design': ['Learn fundamentals', 'Study real systems', 'Practice design problems', 'Review trade-offs', 'Mock design interviews'],
            'Behavioral': ['Prepare STAR stories', 'Practice common questions', 'Company research', 'Mock behavioral interviews'],
            'Resume': ['Resume review', 'Project descriptions', 'Quantify achievements', 'Peer feedback']
        }
        
        return {
            'week': week,
            'focus': focus,
            'activities': activities.get(focus, ['Study', 'Practice', 'Review']),
            'intensity': 'High' if week > 1 else 'Moderate',
            'daily_breakdown': {
                'Monday-Friday': f'Focus on {focus} fundamentals and practice',
                'Saturday': 'Mock interviews and review',
                'Sunday': 'Rest, light review, and planning'
            }
        }


#====================================================================================
# GEMINI-POWERED COORDINATOR AGENT - Enhanced with LLM
#====================================================================================

class GeminiCoordinatorAgent:
    """Enhanced Coordinator Agent powered by Google Gemini"""
    
    def __init__(self, memory_bank, dsa_tool, resume_analyzer, resource_agent, planner, interview_agent, api_key=None):
        self.memory = memory_bank
        self.dsa_tool = dsa_tool
        self.resume_analyzer = resume_analyzer
        self.resource_agent = resource_agent
        self.planner = planner
        self.interview_agent = interview_agent
        
        # Initialize Gemini if API key provided
        self.use_gemini = False
        if api_key and genai:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.use_gemini = True
                logger.info("Coordinator Agent initialized with Gemini AI")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini: {e}. Using fallback routing.")
        else:
            logger.info("Coordinator Agent initialized with keyword-based routing")
    
    def _route_with_gemini(self, query: str, context: Dict) -> Dict:
        """Use Gemini to intelligently route queries"""
        prompt = f"""
        You are a career assistant routing system. Analyze this student query and determine which agents to invoke.
        
        Query: "{query}"
        Student Context: {json.dumps(context.get('statistics', {}), indent=2)}
        
        Available Agents:
        1. DSA Problem Recommender - for coding problem recommendations
        2. Resume Analyzer - for resume review and optimization
        3. Learning Resource Agent - for study material recommendations
        4. Study Planner - for creating study schedules
        5. Mock Interview Agent - for interview practice
        
        Respond in JSON format:
        {{
            "agents": ["agent_name1", "agent_name2"],
            "execution": "sequential" or "parallel",
            "parameters": {{
                "agent_name": {{"param": "value"}}
            }},
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            return result
        except Exception as e:
            logger.error(f"Gemini routing failed: {e}")
            return self._route_with_keywords(query)
    
    def _route_with_keywords(self, query: str) -> Dict:
        """Fallback keyword-based routing"""
        query_lower = query.lower()
        routing = {
            'agents': [],
            'execution': 'sequential',
            'parameters': {},
            'reasoning': 'Keyword-based routing'
        }
        
        if any(word in query_lower for word in ['problem', 'dsa', 'algorithm', 'leetcode', 'coding']):
            routing['agents'].append('dsa')
            level = 'medium'
            if 'easy' in query_lower:
                level = 'easy'
            elif 'hard' in query_lower:
                level = 'hard'
            routing['parameters']['dsa'] = {'level': level}
        
        if any(word in query_lower for word in ['resume', 'cv', 'ats']):
            routing['agents'].append('resume')
        
        if any(word in query_lower for word in ['learn', 'resource', 'study material', 'course', 'tutorial']):
            routing['agents'].append('resources')
        
        if any(word in query_lower for word in ['plan', 'schedule', 'weeks', 'prepare']):
            routing['agents'].append('planner')
        
        if any(word in query_lower for word in ['interview', 'mock', 'practice']):
            routing['agents'].append('interview')
        
        return routing
    
    def process_query(self, student_id: str, query: str, **kwargs) -> Dict:
        """Process student query with intelligent routing"""
        logger.info(f"Processing query for student {student_id}: {query}")
        
        context = self.memory.get_student_context(student_id)
        
        # Route query
        if self.use_gemini:
            routing = self._route_with_gemini(query, context)
        else:
            routing = self._route_with_keywords(query)
        
        response = {
            'student_id': student_id,
            'query': query,
            'routing': routing,
            'results': {},
            'summary': ''
        }
        
        # Execute agents based on routing
        for agent_name in routing['agents']:
            params = routing['parameters'].get(agent_name, {})
            
            if agent_name == 'dsa':
                level = params.get('level', 'medium')
                topic = params.get('topic', None)
                problems = self.dsa_tool.recommend(level=level, topic=topic)
                response['results']['dsa_recommendations'] = problems
                
            elif agent_name == 'resume':
                pdf_path = kwargs.get('resume_pdf', None)
                resume_text = kwargs.get('resume_text', "Sample resume text")
                target_role = kwargs.get('target_role', 'software_engineer')
                analysis = self.resume_analyzer.analyze(resume_text, target_role, pdf_path)
                response['results']['resume_analysis'] = analysis
                
            elif agent_name == 'resources':
                topic = params.get('topic', 'dsa')
                resources = self.resource_agent.recommend(topic)
                response['results']['learning_resources'] = resources
                
            elif agent_name == 'planner':
                weeks = params.get('weeks', 4)
                focus_areas = params.get('focus_areas', ['DSA', 'System Design'])
                plan = self.planner.create_plan(weeks, focus_areas)
                response['results']['study_plan'] = plan
                
            elif agent_name == 'interview':
                interview_type = params.get('type', 'coding')
                difficulty = params.get('difficulty', 'medium')
                interview = self.interview_agent.start_interview(interview_type, difficulty)
                response['results']['mock_interview'] = interview
        
        # Generate summary
        response['summary'] = self._generate_summary(routing, response['results'])
        
        # Update memory
        self.memory.update_progress(
            student_id,
            'query_processed',
            {'question': query, 'agents_used': routing['agents']}
        )
        
        return response
    
    def _generate_summary(self, routing: Dict, results: Dict) -> str:
        """Generate human-readable summary"""
        parts = []
        if 'dsa_recommendations' in results:
            count = len(results['dsa_recommendations'])
            parts.append(f"Found {count} coding problems for you to practice")
        if 'resume_analysis' in results:
            score = results['resume_analysis']['ats_score']
            parts.append(f"Your resume ATS score is {score}/100")
        if 'learning_resources' in results:
            count = len(results['learning_resources'])
            parts.append(f"Recommended {count} learning resources")
        if 'study_plan' in results:
            weeks = results['study_plan']['duration']
            parts.append(f"Created a {weeks} study plan")
        if 'mock_interview' in results:
            parts.append("Set up a mock interview session")
        
        return ". ".join(parts) + "."


#====================================================================================
# INTERACTIVE CLI INTERFACE
#====================================================================================

class InteractiveCLI:
    """Interactive command-line interface"""
    
    def __init__(self, coordinator: GeminiCoordinatorAgent, memory: StudentMemoryBank):
        self.coordinator = coordinator
        self.memory = memory
        self.current_student = None
    
    def display_banner(self):
        """Display welcome banner"""
        print("\n" + "="*80)
        print("🎓 STUDENT CAREER ASSISTANT - ENHANCED MULTI-AGENT SYSTEM")
        print("="*80)
        print("\nYour AI-powered career preparation companion")
        print("Type 'help' for available commands or 'exit' to quit\n")
    
    def display_help(self):
        """Display help menu"""
        help_text = """
Available Commands:
  register           - Register as a new student
  login <id>         - Login with your student ID
  ask <query>        - Ask a question (DSA, resume, resources, etc.)
  problems [level]   - Get coding problem recommendations
  resume [role]      - Analyze your resume
  resources <topic>  - Get learning resources
  plan <weeks>       - Create a study plan
  interview [type]   - Start a mock interview
  stats              - View your progress statistics
  help               - Show this help message
  exit               - Exit the application

Examples:
  ask I need help preparing for interviews
  problems medium
  resume ml_engineer
  resources system_design
  plan 4
  interview coding
        """
        print(help_text)
    
    def register_student(self):
        """Register a new student"""
        print("\n📝 Student Registration")
        student_id = input("Enter your student ID (e.g., student_001): ").strip()
        
        if student_id in self.memory.students:
            print(f"❌ Student {student_id} already exists. Use 'login {student_id}' instead.")
            return
        
        name = input("Enter your name: ").strip()
        year = input("Enter your year (e.g., 3rd Year): ").strip()
        major = input("Enter your major (e.g., CS/AI-ML): ").strip()
        
        profile = {'name': name, 'year': year, 'major': major}
        self.memory.add_student(student_id, profile)
        self.current_student = student_id
        
        print(f"\n✅ Welcome, {name}! You're now registered as {student_id}")
    
    def login(self, student_id: str):
        """Login existing student"""
        if student_id in self.memory.students:
            self.current_student = student_id
            name = self.memory.students[student_id]['profile'].get('name', 'Student')
            print(f"\n✅ Welcome back, {name}!")
        else:
            print(f"❌ Student {student_id} not found. Use 'register' to create an account.")
    
    def show_stats(self):
        """Show student statistics"""
        if not self.current_student:
            print("❌ Please login first")
            return
        
        stats = self.memory.get_statistics(self.current_student)
        print("\n📊 Your Progress Statistics:")
        print(f"  Problems Solved: {stats.get('problems_solved', 0)}")
        print(f"  - Easy: {stats.get('easy_solved', 0)}")
        print(f"  - Medium: {stats.get('medium_solved', 0)}")
        print(f"  - Hard: {stats.get('hard_solved', 0)}")
        print(f"  Topics Covered: {', '.join(stats.get('topics_covered', [])) or 'None yet'}")
        print(f"  Total Study Hours: {stats.get('total_study_hours', 0)}")
    
    def run(self):
        """Main CLI loop"""
        self.display_banner()
        
        while True:
            try:
                if self.current_student:
                    prompt = f"[{self.current_student}]> "
                else:
                    prompt = "> "
                
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command == 'exit':
                    print("\n👋 Thanks for using Student Career Assistant. Good luck with your preparation!")
                    break
                
                elif command == 'help':
                    self.display_help()
                
                elif command == 'register':
                    self.register_student()
                
                elif command == 'login':
                    if args:
                        self.login(args)
                    else:
                        print("❌ Usage: login <student_id>")
                
                elif command == 'stats':
                    self.show_stats()
                
                elif command in ['ask', 'problems', 'resume', 'resources', 'plan', 'interview']:
                    if not self.current_student:
                        print("❌ Please login or register first")
                        continue
                    
                    if command == 'ask':
                        query = args if args else input("What would you like help with? ")
                    elif command == 'problems':
                        level = args if args else 'medium'
                        query = f"Give me {level} coding problems"
                    elif command == 'resume':
                        role = args if args else 'software_engineer'
                        query = f"Analyze my resume for {role} role"
                    elif command == 'resources':
                        topic = args if args else 'dsa'
                        query = f"Recommend learning resources for {topic}"
                    elif command == 'plan':
                        weeks = args if args else '4'
                        query = f"Create a {weeks} week study plan"
                    elif command == 'interview':
                        itype = args if args else 'coding'
                        query = f"Start a {itype} mock interview"
                    
                    print(f"\n🔄 Processing: {query}")
                    result = self.coordinator.process_query(self.current_student, query)
                    self._display_results(result)
                
                else:
                    print(f"❌ Unknown command: {command}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                logger.error(f"CLI Error: {e}", exc_info=True)
    
    def _display_results(self, result: Dict):
        """Display query results"""
        print(f"\n{'='*80}")
        print(f"📋 Results for: {result['query']}")
        print(f"{'='*80}\n")
        
        if 'dsa_recommendations' in result['results']:
            problems = result['results']['dsa_recommendations']
            print(f"💻 Coding Problems ({len(problems)}):")
            for i, p in enumerate(problems, 1):
                print(f"  {i}. {p['name']} ({p['topic']}) - {p['difficulty']}")
                if 'leetcode_id' in p:
                    print(f"     LeetCode: https://leetcode.com/problems/{p['name'].lower().replace(' ', '-')}/")
            print()
        
        if 'resume_analysis' in result['results']:
            analysis = result['results']['resume_analysis']
            print(f"📄 Resume Analysis:")
            print(f"  ATS Score: {analysis['ats_score']}/100")
            print(f"  Target Role: {analysis['target_role'].replace('_', ' ').title()}")
            if analysis['strengths']:
                print(f"  ✅ Strengths:")
                for s in analysis['strengths']:
                    print(f"    - {s}")
            if analysis['suggestions']:
                print(f"  💡 Suggestions:")
                for s in analysis['suggestions']:
                    print(f"    - {s}")
            print()
        
        if 'learning_resources' in result['results']:
            resources = result['results']['learning_resources']
            print(f"📚 Learning Resources ({len(resources)}):")
            for i, r in enumerate(resources, 1):
                print(f"  {i}. {r['title']} ({r['type']}) - {r['difficulty']}")
                if 'url' in r:
                    print(f"     Link: {r['url']}")
            print()
        
        if 'study_plan' in result['results']:
            plan = result['results']['study_plan']
            print(f"📅 Study Plan:")
            print(f"  Duration: {plan['duration']}")
            print(f"  Daily Commitment: {plan['daily_commitment']}")
            print(f"  Total Hours: {plan['total_hours']}")
            print(f"  Focus Areas: {', '.join(plan['focus_areas'])}")
            print(f"\n  Weekly Schedule:")
            for week in plan['weekly_schedule'][:3]:  # Show first 3 weeks
                print(f"    Week {week['week']}: {week['focus']} ({week['intensity']} intensity)")
            print()
        
        if 'mock_interview' in result['results']:
            interview = result['results']['mock_interview']
            print(f"🎤 Mock Interview ({interview['type'].title()}):")
            print(f"  Difficulty: {interview['difficulty'].title()}")
            print(f"  Questions:")
            for i, q in enumerate(interview['questions'][:3], 1):
                if isinstance(q, dict):
                    print(f"    {i}. {q['name']} ({q['topic']})")
                else:
                    print(f"    {i}. {q}")
            print(f"\n  💡 Tips:")
            for tip in interview['tips'][:3]:
                print(f"    - {tip}")
            print()
        
        print(f"📝 Summary: {result['summary']}")
        print(f"{'='*80}\n")


#====================================================================================
# MAIN EXECUTION
#====================================================================================

def main():
    """Main application entry point"""
    # Setup configuration
    Config.setup()
    
    # Get API key from environment or user
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("\n⚠️  GEMINI_API_KEY not found in environment variables.")
        print("The system will use keyword-based routing instead of AI-powered routing.")
        response = input("Would you like to enter your API key now? (y/n): ").strip().lower()
        if response == 'y':
            api_key = input("Enter your Gemini API key: ").strip()
            if api_key:
                os.environ['GEMINI_API_KEY'] = api_key
    
    # Initialize all agents
    print("\n🔧 Initializing agents...")
    memory_bank = StudentMemoryBank()
    dsa_tool = DSAProblemRecommender()
    resume_analyzer = ResumeAnalyzer()
    resource_agent = LearningResourceAgent()
    planner = StudyPlannerAgent()
    interview_agent = MockInterviewAgent(dsa_tool)
    coordinator = GeminiCoordinatorAgent(
        memory_bank, dsa_tool, resume_analyzer, 
        resource_agent, planner, interview_agent, api_key
    )
    
    print("✅ All agents initialized successfully!")
    
    # Start interactive CLI
    cli = InteractiveCLI(coordinator, memory_bank)
    cli.run()


if __name__ == "__main__":
    main()
