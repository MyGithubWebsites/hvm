#!/usr/bin/env python3
"""
Comprehensive Test Suite for Student Career Assistant Agent
Tests all agent functionality, memory management, and integrations
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import all components from enhanced_agent
import sys
sys.path.insert(0, str(Path(__file__).parent))

from agent import (
    Config,
    StudentMemoryBank,
    DSAProblemRecommender,
    ResumeAnalyzer,
    MockInterviewAgent,
    LearningResourceAgent,
    StudyPlannerAgent,
    GeminiCoordinatorAgent
)


#====================================================================================
# FIXTURES
#====================================================================================

@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing"""
    temp_dir = Path(tempfile.mkdtemp())
    original_data_dir = Config.DATA_DIR
    Config.DATA_DIR = temp_dir
    Config.STUDENTS_FILE = temp_dir / "students.json"
    Config.PROBLEMS_FILE = temp_dir / "problems.json"
    Config.setup()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)
    Config.DATA_DIR = original_data_dir


@pytest.fixture
def memory_bank(temp_data_dir):
    """Create a fresh memory bank for testing"""
    return StudentMemoryBank()


@pytest.fixture
def dsa_tool():
    """Create DSA Problem Recommender"""
    return DSAProblemRecommender()


@pytest.fixture
def resume_analyzer():
    """Create Resume Analyzer"""
    return ResumeAnalyzer()


@pytest.fixture
def interview_agent(dsa_tool):
    """Create Mock Interview Agent"""
    return MockInterviewAgent(dsa_tool)


@pytest.fixture
def resource_agent():
    """Create Learning Resource Agent"""
    return LearningResourceAgent()


@pytest.fixture
def planner():
    """Create Study Planner Agent"""
    return StudyPlannerAgent()


@pytest.fixture
def coordinator(memory_bank, dsa_tool, resume_analyzer, resource_agent, planner, interview_agent):
    """Create Coordinator Agent without Gemini API"""
    return GeminiCoordinatorAgent(
        memory_bank, dsa_tool, resume_analyzer, 
        resource_agent, planner, interview_agent, 
        api_key=None
    )


#====================================================================================
# MEMORY BANK TESTS
#====================================================================================

class TestStudentMemoryBank:
    """Test StudentMemoryBank functionality"""
    
    def test_initialization(self, memory_bank):
        """Test memory bank initializes correctly"""
        assert isinstance(memory_bank.students, dict)
        assert len(memory_bank.students) == 0
    
    def test_add_student(self, memory_bank):
        """Test adding a new student"""
        profile = {'name': 'John Doe', 'year': '3rd Year', 'major': 'CS'}
        memory_bank.add_student('test_001', profile)
        
        assert 'test_001' in memory_bank.students
        assert memory_bank.students['test_001']['profile'] == profile
        assert 'statistics' in memory_bank.students['test_001']
        assert memory_bank.students['test_001']['statistics']['problems_solved'] == 0
    
    def test_persistence(self, temp_data_dir):
        """Test student data persists across instances"""
        # Create first instance and add student
        bank1 = StudentMemoryBank()
        bank1.add_student('test_001', {'name': 'Alice'})
        
        # Create second instance and verify data persists
        bank2 = StudentMemoryBank()
        assert 'test_001' in bank2.students
        assert bank2.students['test_001']['profile']['name'] == 'Alice'
    
    def test_update_progress(self, memory_bank):
        """Test updating student progress"""
        memory_bank.add_student('test_001', {'name': 'Bob'})
        memory_bank.update_progress('test_001', 'solved_problem', {'problem': 'Two Sum'})
        
        progress = memory_bank.students['test_001']['progress']
        assert len(progress) == 1
        assert progress[0]['activity'] == 'solved_problem'
        assert progress[0]['details']['problem'] == 'Two Sum'
    
    def test_update_statistics(self, memory_bank):
        """Test updating student statistics"""
        memory_bank.add_student('test_001', {'name': 'Charlie'})
        memory_bank.update_statistics('test_001', 'easy_solved', 1)
        memory_bank.update_statistics('test_001', 'problems_solved', 1)
        memory_bank.update_statistics('test_001', 'topics_covered', 'Array')
        
        stats = memory_bank.get_statistics('test_001')
        assert stats['easy_solved'] == 1
        assert stats['problems_solved'] == 1
        assert 'Array' in stats['topics_covered']
    
    def test_get_student_context(self, memory_bank):
        """Test retrieving student context"""
        profile = {'name': 'Dave', 'major': 'AI-ML'}
        memory_bank.add_student('test_001', profile)
        
        context = memory_bank.get_student_context('test_001')
        assert context['profile'] == profile
        assert 'statistics' in context
        assert 'progress' in context


#====================================================================================
# DSA PROBLEM RECOMMENDER TESTS
#====================================================================================

class TestDSAProblemRecommender:
    """Test DSA Problem Recommender functionality"""
    
    def test_initialization(self, dsa_tool):
        """Test DSA tool initializes with problems"""
        assert 'easy' in dsa_tool.problems
        assert 'medium' in dsa_tool.problems
        assert 'hard' in dsa_tool.problems
        assert len(dsa_tool.problems['easy']) > 0
    
    def test_recommend_by_level(self, dsa_tool):
        """Test recommending problems by difficulty level"""
        easy_problems = dsa_tool.recommend(level='easy', count=5)
        assert len(easy_problems) <= 5
        assert all(p['difficulty'] == 'Easy' for p in easy_problems)
        
        hard_problems = dsa_tool.recommend(level='hard', count=3)
        assert len(hard_problems) <= 3
        assert all(p['difficulty'] == 'Hard' for p in hard_problems)
    
    def test_recommend_by_topic(self, dsa_tool):
        """Test recommending problems by topic"""
        array_problems = dsa_tool.recommend(level='easy', topic='Array')
        assert all(p['topic'] == 'Array' for p in array_problems)
    
    def test_get_topics(self, dsa_tool):
        """Test getting available topics"""
        all_topics = dsa_tool.get_topics()
        assert isinstance(all_topics, list)
        assert 'Array' in all_topics
        
        easy_topics = dsa_tool.get_topics(level='easy')
        assert isinstance(easy_topics, list)
    
    def test_problem_structure(self, dsa_tool):
        """Test that problems have required fields"""
        problems = dsa_tool.recommend(level='medium', count=1)
        if problems:
            problem = problems[0]
            assert 'name' in problem
            assert 'topic' in problem
            assert 'difficulty' in problem
            assert 'leetcode_id' in problem


#====================================================================================
# RESUME ANALYZER TESTS
#====================================================================================

class TestResumeAnalyzer:
    """Test Resume Analyzer functionality"""
    
    def test_initialization(self, resume_analyzer):
        """Test resume analyzer initializes with keywords"""
        assert 'software_engineer' in resume_analyzer.ats_keywords
        assert 'ml_engineer' in resume_analyzer.ats_keywords
        assert len(resume_analyzer.action_verbs) > 0
    
    def test_basic_analysis(self, resume_analyzer):
        """Test basic resume analysis"""
        resume_text = """
        Software Engineer with 3 years of experience.
        Developed web applications using Python, JavaScript, and React.
        Implemented RESTful APIs and optimized database queries.
        Experience with Docker, Git, and Agile methodologies.
        """
        
        analysis = resume_analyzer.analyze(resume_text, target_role='software_engineer')
        
        assert 'ats_score' in analysis
        assert analysis['ats_score'] > 0
        assert isinstance(analysis['keyword_analysis'], dict)
        assert isinstance(analysis['suggestions'], list)
        assert isinstance(analysis['strengths'], list)
    
    def test_keyword_detection(self, resume_analyzer):
        """Test keyword detection in resume"""
        resume_text = "Python developer with experience in tensorflow, pytorch, and pandas"
        analysis = resume_analyzer.analyze(resume_text, target_role='ml_engineer')
        
        found_keywords = analysis['keyword_analysis']['found']
        assert 'python' in found_keywords
        assert 'tensorflow' in found_keywords
        assert 'pytorch' in found_keywords
    
    def test_action_verb_detection(self, resume_analyzer):
        """Test action verb detection"""
        resume_text = "Developed, implemented, and optimized various systems"
        analysis = resume_analyzer.analyze(resume_text)
        
        action_verbs = analysis['action_verbs_found']
        assert 'developed' in action_verbs
        assert 'implemented' in action_verbs
        assert 'optimized' in action_verbs
    
    def test_ats_score_calculation(self, resume_analyzer):
        """Test ATS score calculation"""
        # High-quality resume
        good_resume = """
        Senior Software Engineer with 5 years of experience.
        Developed web applications using Python, Java, JavaScript, React, Node.js, and SQL.
        Implemented microservices architecture using Docker and deployed to production.
        Led agile team and optimized API performance by 40%.
        Experience with Git, CI/CD, and test-driven development.
        """
        
        # Poor-quality resume
        poor_resume = "I like computers and programming."
        
        good_analysis = resume_analyzer.analyze(good_resume, target_role='software_engineer')
        poor_analysis = resume_analyzer.analyze(poor_resume, target_role='software_engineer')
        
        assert good_analysis['ats_score'] > poor_analysis['ats_score']
    
    def test_role_specific_analysis(self, resume_analyzer):
        """Test role-specific keyword analysis"""
        ml_resume = "Python developer with TensorFlow, PyTorch, and deep learning experience"
        
        ml_analysis = resume_analyzer.analyze(ml_resume, target_role='ml_engineer')
        swe_analysis = resume_analyzer.analyze(ml_resume, target_role='software_engineer')
        
        # ML resume should score higher for ML role
        assert ml_analysis['ats_score'] >= swe_analysis['ats_score']


#====================================================================================
# MOCK INTERVIEW AGENT TESTS
#====================================================================================

class TestMockInterviewAgent:
    """Test Mock Interview Agent functionality"""
    
    def test_initialization(self, interview_agent):
        """Test interview agent initializes correctly"""
        assert interview_agent.problem_recommender is not None
        assert 'behavioral' in interview_agent.interview_questions
        assert 'system_design' in interview_agent.interview_questions
    
    def test_coding_interview(self, interview_agent):
        """Test starting a coding interview"""
        interview = interview_agent.start_interview(interview_type='coding', difficulty='medium')
        
        assert interview['type'] == 'coding'
        assert interview['difficulty'] == 'medium'
        assert len(interview['questions']) > 0
        assert len(interview['tips']) > 0
        assert all(isinstance(q, dict) for q in interview['questions'])
    
    def test_behavioral_interview(self, interview_agent):
        """Test starting a behavioral interview"""
        interview = interview_agent.start_interview(interview_type='behavioral')
        
        assert interview['type'] == 'behavioral'
        assert len(interview['questions']) > 0
        assert len(interview['tips']) > 0
        assert all(isinstance(q, str) for q in interview['questions'])
    
    def test_system_design_interview(self, interview_agent):
        """Test starting a system design interview"""
        interview = interview_agent.start_interview(interview_type='system_design')
        
        assert interview['type'] == 'system_design'
        assert len(interview['questions']) > 0
        assert len(interview['tips']) > 0
    
    def test_evaluate_answer(self, interview_agent):
        """Test answer evaluation"""
        evaluation = interview_agent.evaluate_answer(
            "Two Sum problem",
            "I would use a hashmap to store complements"
        )
        
        assert 'question' in evaluation
        assert 'feedback' in evaluation
        assert 'score' in evaluation
        assert 'improvement_areas' in evaluation


#====================================================================================
# LEARNING RESOURCE AGENT TESTS
#====================================================================================

class TestLearningResourceAgent:
    """Test Learning Resource Agent functionality"""
    
    def test_initialization(self, resource_agent):
        """Test resource agent initializes with resources"""
        assert 'dsa' in resource_agent.resources
        assert 'system_design' in resource_agent.resources
        assert 'machine_learning' in resource_agent.resources
    
    def test_recommend_resources(self, resource_agent):
        """Test recommending resources"""
        dsa_resources = resource_agent.recommend('dsa', count=3)
        
        assert len(dsa_resources) <= 3
        assert all('title' in r for r in dsa_resources)
        assert all('type' in r for r in dsa_resources)
        assert all('difficulty' in r for r in dsa_resources)
    
    def test_get_all_topics(self, resource_agent):
        """Test getting all available topics"""
        topics = resource_agent.get_all_topics()
        
        assert isinstance(topics, list)
        assert 'dsa' in topics
        assert 'system_design' in topics
    
    def test_filter_by_level(self, resource_agent):
        """Test filtering resources by difficulty level"""
        beginner_resources = resource_agent.recommend('dsa', level='Beginner')
        
        assert all(
            r['difficulty'] in ['Beginner', 'All levels'] 
            for r in beginner_resources
        )


#====================================================================================
# STUDY PLANNER AGENT TESTS
#====================================================================================

class TestStudyPlannerAgent:
    """Test Study Planner Agent functionality"""
    
    def test_initialization(self, planner):
        """Test planner initializes correctly"""
        assert planner is not None
    
    def test_create_basic_plan(self, planner):
        """Test creating a basic study plan"""
        plan = planner.create_plan(
            weeks=4,
            focus_areas=['DSA', 'System Design'],
            hours_per_day=4
        )
        
        assert plan['duration'] == '4 weeks'
        assert plan['daily_commitment'] == '4 hours/day'
        assert plan['total_hours'] == 4 * 7 * 4
        assert len(plan['weekly_schedule']) == 4
        assert len(plan['milestones']) == 4
    
    def test_daily_routine_generation(self, planner):
        """Test daily routine generation"""
        # Test 2-hour plan
        plan_2h = planner.create_plan(weeks=1, focus_areas=['DSA'], hours_per_day=2)
        assert 'learning' in plan_2h['daily_routine']
        
        # Test 4-hour plan
        plan_4h = planner.create_plan(weeks=1, focus_areas=['DSA'], hours_per_day=4)
        assert 'practice' in plan_4h['daily_routine']
        
        # Test 6-hour plan
        plan_6h = planner.create_plan(weeks=1, focus_areas=['DSA'], hours_per_day=6)
        assert 'mock_interview' in plan_6h['daily_routine']
    
    def test_week_plan_structure(self, planner):
        """Test weekly plan structure"""
        plan = planner.create_plan(weeks=2, focus_areas=['DSA'], hours_per_day=4)
        
        week_plan = plan['weekly_schedule'][0]
        assert 'week' in week_plan
        assert 'focus' in week_plan
        assert 'activities' in week_plan
        assert 'intensity' in week_plan
        assert 'daily_breakdown' in week_plan
    
    def test_milestone_generation(self, planner):
        """Test milestone generation"""
        plan = planner.create_plan(weeks=3, focus_areas=['DSA'], hours_per_day=4)
        
        milestones = plan['milestones']
        assert len(milestones) == 3
        assert all('week' in m for m in milestones)
        assert all('goal' in m for m in milestones)
        assert all('assessment' in m for m in milestones)


#====================================================================================
# COORDINATOR AGENT TESTS
#====================================================================================

class TestGeminiCoordinatorAgent:
    """Test Coordinator Agent functionality"""
    
    def test_initialization(self, coordinator):
        """Test coordinator initializes correctly"""
        assert coordinator.memory is not None
        assert coordinator.dsa_tool is not None
        assert coordinator.resume_analyzer is not None
        assert coordinator.resource_agent is not None
        assert coordinator.planner is not None
        assert coordinator.interview_agent is not None
    
    def test_keyword_routing(self, coordinator, memory_bank):
        """Test keyword-based query routing"""
        memory_bank.add_student('test_001', {'name': 'Test User'})
        
        # Test DSA routing
        dsa_routing = coordinator._route_with_keywords("Give me medium coding problems")
        assert 'dsa' in dsa_routing['agents']
        
        # Test resume routing
        resume_routing = coordinator._route_with_keywords("Review my resume")
        assert 'resume' in resume_routing['agents']
        
        # Test planner routing
        plan_routing = coordinator._route_with_keywords("Create a 4 week study plan")
        assert 'planner' in plan_routing['agents']
    
    def test_process_dsa_query(self, coordinator, memory_bank):
        """Test processing DSA query"""
        memory_bank.add_student('test_001', {'name': 'Test User'})
        
        result = coordinator.process_query(
            'test_001',
            'Give me easy coding problems'
        )
        
        assert 'dsa_recommendations' in result['results']
        assert len(result['results']['dsa_recommendations']) > 0
        assert result['query'] == 'Give me easy coding problems'
    
    def test_process_resume_query(self, coordinator, memory_bank):
        """Test processing resume query"""
        memory_bank.add_student('test_001', {'name': 'Test User'})
        
        result = coordinator.process_query(
            'test_001',
            'Analyze my resume',
            resume_text='Software engineer with Python and Java experience'
        )
        
        assert 'resume_analysis' in result['results']
        assert 'ats_score' in result['results']['resume_analysis']
    
    def test_process_multi_agent_query(self, coordinator, memory_bank):
        """Test processing query requiring multiple agents"""
        memory_bank.add_student('test_001', {'name': 'Test User'})
        
        result = coordinator.process_query(
            'test_001',
            'Help me prepare for interviews with problems and resources'
        )
        
        # Should invoke both DSA and Resources agents
        results = result['results']
        assert len(results) >= 1  # At least one agent should respond
    
    def test_summary_generation(self, coordinator):
        """Test summary generation"""
        routing = {'agents': ['dsa', 'resume']}
        results = {
            'dsa_recommendations': [{'name': 'Two Sum'}],
            'resume_analysis': {'ats_score': 75}
        }
        
        summary = coordinator._generate_summary(routing, results)
        
        assert 'coding problems' in summary.lower()
        assert '75' in summary
    
    def test_memory_update(self, coordinator, memory_bank):
        """Test that queries update student memory"""
        memory_bank.add_student('test_001', {'name': 'Test User'})
        
        coordinator.process_query('test_001', 'Give me problems')
        
        context = memory_bank.get_student_context('test_001')
        assert len(context['progress']) > 0
        assert context['progress'][0]['activity'] == 'query_processed'


#====================================================================================
# INTEGRATION TESTS
#====================================================================================

class TestIntegration:
    """Integration tests for the entire system"""
    
    def test_end_to_end_workflow(self, temp_data_dir):
        """Test complete end-to-end workflow"""
        # Initialize all components
        memory = StudentMemoryBank()
        dsa = DSAProblemRecommender()
        resume = ResumeAnalyzer()
        resources = LearningResourceAgent()
        planner = StudyPlannerAgent()
        interview = MockInterviewAgent(dsa)
        coord = GeminiCoordinatorAgent(
            memory, dsa, resume, resources, planner, interview, None
        )
        
        # Add student
        memory.add_student('student_001', {
            'name': 'Alice',
            'year': '3rd Year',
            'major': 'CS'
        })
        
        # Process various queries
        result1 = coord.process_query('student_001', 'Give me coding problems')
        assert 'dsa_recommendations' in result1['results']
        
        result2 = coord.process_query('student_001', 'Create a study plan for 2 weeks')
        assert 'study_plan' in result2['results']
        
        result3 = coord.process_query('student_001', 'Start a mock interview')
        assert 'mock_interview' in result3['results']
        
        # Verify progress tracking
        context = memory.get_student_context('student_001')
        assert len(context['progress']) >= 3


#====================================================================================
# RUN TESTS
#====================================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
