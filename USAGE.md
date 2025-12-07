# 📖 Student Career Assistant - Usage Guide

Complete guide to using the Enhanced Student Career Assistant Agent system.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd student-career-assistant-agent

# Install dependencies
pip install -r requirements.txt

# Install dependencies (if in virtual environment)
pip install google-generativeai PyPDF2 python-dateutil


# Optional: Set up Gemini API key for AI-powered routing
export GEMINI_API_KEY='your-api-key-here'
```

### 2. Run the Enhanced Agent

```bash
# Run the enhanced interactive version
python agent.py

```

## 🎯 Interactive CLI Commands

### Basic Commands

| Command | Description | Example |
|---------|-------------|---------|
| `register` | Register as a new student | `register` |
| `login <id>` | Login with student ID | `login student_001` |
| `help` | Show all available commands | `help` |
| `exit` | Exit the application | `exit` |
| `stats` | View your progress statistics | `stats` |

### Career Assistance Commands

| Command | Description | Example |
|---------|-------------|---------|
| `ask <query>` | Ask any career-related question | `ask I need help preparing for interviews` |
| `problems [level]` | Get coding problem recommendations | `problems medium` |
| `resume [role]` | Analyze your resume | `resume ml_engineer` |
| `resources <topic>` | Get learning resources | `resources system_design` |
| `plan <weeks>` | Create a study plan | `plan 4` |
| `interview [type]` | Start a mock interview | `interview coding` |

## 💡 Usage Examples

### Example 1: Complete Interview Preparation Workflow

```
> register
Enter your student ID: student_123
Enter your name: Alice Johnson
Enter your year: 3rd Year
Enter your major: Computer Science

[student_123]> ask I have an interview in 3 weeks for a software engineer role

🔄 Processing: I have an interview in 3 weeks for a software engineer role

================================================================================
📋 Results for: I have an interview in 3 weeks for a software engineer role
================================================================================

💻 Coding Problems (5):
  1. LRU Cache (Design) - Medium
     LeetCode: https://leetcode.com/problems/lru-cache/
  2. Binary Tree Level Order Traversal (Tree) - Medium
     LeetCode: https://leetcode.com/problems/binary-tree-level-order-traversal/
  3. Longest Substring Without Repeating Characters (String) - Medium
     LeetCode: https://leetcode.com/problems/longest-substring-without-repeating-characters/
  ...

📚 Learning Resources (5):
  1. NeetCode.io - Curated LeetCode Problems (Practice) - All levels
     Link: https://neetcode.io
  2. LeetCode Patterns by Sean Prashad (Guide) - All levels
     Link: https://seanprashad.com/leetcode-patterns/
  ...

📅 Study Plan:
  Duration: 4 weeks
  Daily Commitment: 4 hours/day
  Total Hours: 112
  Focus Areas: DSA, System Design
  
  Weekly Schedule:
    Week 1: DSA (Moderate intensity)
    Week 2: System Design (High intensity)
    Week 3: DSA (High intensity)

📝 Summary: Found 5 coding problems for you to practice. Recommended 5 learning resources. Created a 4 weeks study plan.
================================================================================
```

### Example 2: Resume Analysis

```
[student_123]> resume software_engineer

🔄 Processing: Analyze my resume for software_engineer role

================================================================================
📋 Results for: Analyze my resume for software_engineer role
================================================================================

📄 Resume Analysis:
  ATS Score: 65/100
  Target Role: Software Engineer
  ✅ Strengths:
    - Strong technical keyword coverage (7/10)
    - Good use of action verbs (6 found)
  💡 Suggestions:
    - Add these Software Engineer keywords: node, api, docker
    - Add quantifiable metrics (e.g., 'Improved performance by 40%', 'Reduced latency by 2s')

📝 Summary: Your resume ATS score is 65/100
================================================================================
```

### Example 3: Mock Interview Practice

```
[student_123]> interview coding

🔄 Processing: Start a coding mock interview

================================================================================
📋 Results for: Start a coding mock interview
================================================================================

🎤 Mock Interview (Coding):
  Difficulty: Medium
  Questions:
    1. Three Sum (Array)
    2. Longest Substring Without Repeating Characters (String)
    3. Binary Tree Level Order Traversal (Tree)
  
  💡 Tips:
    - Think out loud - explain your approach before coding
    - Ask clarifying questions about edge cases
    - Discuss time and space complexity

📝 Summary: Set up a mock interview session.
================================================================================
```

### Example 4: Learning Resources for Specific Topics

```
[student_123]> resources machine_learning

🔄 Processing: Recommend learning resources for machine_learning

================================================================================
📋 Results for: Recommend learning resources for machine_learning
================================================================================

📚 Learning Resources (3):
  1. Fast.ai Practical Deep Learning (Course) - Beginner
     Link: https://fast.ai
  2. Andrew Ng Machine Learning Course (Course) - Beginner
     Link: https://coursera.org
  3. Kaggle Learn (Practice) - All levels
     Link: https://kaggle.com/learn

📝 Summary: Recommended 3 learning resources.
================================================================================
```

### Example 5: Viewing Progress Statistics

```
[student_123]> stats

📊 Your Progress Statistics:
  Problems Solved: 0
  - Easy: 0
  - Medium: 0
  - Hard: 0
  Topics Covered: None yet
  Total Study Hours: 0
```

## 🔧 Programmatic Usage

### Python API Examples

#### Example 1: Initialize the System

```python
from enhanced_agent import (
    Config,
    StudentMemoryBank,
    DSAProblemRecommender,
    ResumeAnalyzer,
    LearningResourceAgent,
    StudyPlannerAgent,
    MockInterviewAgent,
    GeminiCoordinatorAgent
)

# Setup configuration
Config.setup()

# Initialize all agents
memory_bank = StudentMemoryBank()
dsa_tool = DSAProblemRecommender()
resume_analyzer = ResumeAnalyzer()
resource_agent = LearningResourceAgent()
planner = StudyPlannerAgent()
interview_agent = MockInterviewAgent(dsa_tool)

# Create coordinator
coordinator = GeminiCoordinatorAgent(
    memory_bank, dsa_tool, resume_analyzer,
    resource_agent, planner, interview_agent,
    api_key='your-gemini-api-key'  # Optional
)
```

#### Example 2: Register a Student

```python
# Add a new student
memory_bank.add_student('student_001', {
    'name': 'John Doe',
    'year': '4th Year',
    'major': 'Computer Science'
})

# Update progress
memory_bank.update_progress('student_001', 'solved_problem', {
    'problem': 'Two Sum',
    'difficulty': 'Easy',
    'time_taken': '15 minutes'
})

# Update statistics
memory_bank.update_statistics('student_001', 'easy_solved', 1)
memory_bank.update_statistics('student_001', 'problems_solved', 1)
memory_bank.update_statistics('student_001', 'topics_covered', 'Array')
```

#### Example 3: Get Problem Recommendations

```python
# Get medium-level problems
problems = dsa_tool.recommend(level='medium', count=5)

# Get array-specific problems
array_problems = dsa_tool.recommend(level='easy', topic='Array', count=3)

# Get all available topics
topics = dsa_tool.get_topics()
print(f"Available topics: {topics}")
```

#### Example 4: Analyze a Resume

```python
resume_text = """
Senior Software Engineer with 5 years of experience.
Developed scalable web applications using Python, JavaScript, React, and Node.js.
Implemented microservices architecture and optimized database queries.
Experience with Docker, Kubernetes, AWS, and CI/CD pipelines.
Led a team of 5 engineers and improved system performance by 40%.
"""

analysis = resume_analyzer.analyze(
    resume_text,
    target_role='software_engineer'
)

print(f"ATS Score: {analysis['ats_score']}/100")
print(f"Strengths: {analysis['strengths']}")
print(f"Suggestions: {analysis['suggestions']}")
```

#### Example 5: Create a Study Plan

```python
# Create a 4-week study plan
plan = planner.create_plan(
    weeks=4,
    focus_areas=['DSA', 'System Design', 'Behavioral'],
    hours_per_day=5,
    skill_level='intermediate'
)

print(f"Duration: {plan['duration']}")
print(f"Total Hours: {plan['total_hours']}")
print(f"Daily Routine: {plan['daily_routine']}")

for week in plan['weekly_schedule']:
    print(f"\nWeek {week['week']}: {week['focus']}")
    print(f"Activities: {', '.join(week['activities'])}")
```

#### Example 6: Start a Mock Interview

```python
# Coding interview
coding_interview = interview_agent.start_interview(
    interview_type='coding',
    difficulty='medium'
)

print("Coding Interview Questions:")
for i, problem in enumerate(coding_interview['questions'], 1):
    print(f"{i}. {problem['name']} ({problem['topic']})")

# Behavioral interview
behavioral_interview = interview_agent.start_interview(
    interview_type='behavioral'
)

print("\nBehavioral Interview Questions:")
for i, question in enumerate(behavioral_interview['questions'], 1):
    print(f"{i}. {question}")

# System design interview
design_interview = interview_agent.start_interview(
    interview_type='system_design'
)

print("\nSystem Design Question:")
print(design_interview['questions'][0])
```

#### Example 7: Use the Coordinator Agent

```python
# Process a comprehensive query
result = coordinator.process_query(
    student_id='student_001',
    query='I need help preparing for my Google interview in 2 weeks'
)

# Access results
if 'dsa_recommendations' in result['results']:
    problems = result['results']['dsa_recommendations']
    print(f"Recommended {len(problems)} problems")

if 'study_plan' in result['results']:
    plan = result['results']['study_plan']
    print(f"Created {plan['duration']} study plan")

print(f"\nSummary: {result['summary']}")
```

## 🧪 Running Tests

```bash
# Run all tests
pytest test_agents.py -v

# Run specific test class
pytest test_agents.py::TestDSAProblemRecommender -v

# Run with coverage
pytest test_agents.py --cov=enhanced_agent --cov-report=html

# Run tests in parallel
pytest test_agents.py -n auto
```

## 📊 Data Persistence

Student data is automatically saved to JSON files in the `data/` directory:

```
data/
├── students.json      # Student profiles and progress
└── problems.json      # (Reserved for future use)
```

### Student Data Structure

```json
{
  "student_001": {
    "profile": {
      "name": "John Doe",
      "year": "4th Year",
      "major": "Computer Science"
    },
    "progress": [
      {
        "activity": "query_processed",
        "details": { "question": "..." },
        "timestamp": "2025-11-15T10:30:00"
      }
    ],
    "statistics": {
      "problems_solved": 15,
      "easy_solved": 5,
      "medium_solved": 8,
      "hard_solved": 2,
      "topics_covered": ["Array", "Tree", "Graph"],
      "total_study_hours": 40
    },
    "created_at": "2025-11-10T09:00:00",
    "last_active": "2025-11-15T10:30:00"
  }
}
```

## 🎛️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GEMINI_API_KEY` | Google Gemini API key for AI-powered routing | No | None (uses keyword routing) |

### Setting API Key

```bash
# Linux/Mac
export GEMINI_API_KEY='your-api-key-here'

# Windows (Command Prompt)
set GEMINI_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:GEMINI_API_KEY='your-api-key-here'

# Or add to ~/.bashrc or ~/.zshrc for persistence
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## 🐛 Troubleshooting

### Issue: "google-generativeai not installed"

```bash
pip install google-generativeai
```

### Issue: "PyPDF2 not installed"

```bash
pip install PyPDF2
```

### Issue: Data directory not found

The system automatically creates the `data/` directory on first run. If you encounter issues:

```python
from enhanced_agent import Config
Config.setup()
```

### Issue: API key not recognized

Make sure the environment variable is set in the same terminal session where you run the script:

```bash
echo $GEMINI_API_KEY  # Should print your API key
```

### Issue: Permission denied when writing to data directory

```bash
chmod +w data/
chmod +w data/students.json
```

## 📈 Performance Tips

1. **Use keyword routing for faster responses** - If you don't need AI-powered routing, skip the GEMINI_API_KEY
2. **Batch operations** - When using the API programmatically, process multiple students in a loop
3. **Regular data cleanup** - Periodically archive old student data to keep the JSON files manageable
4. **Logging** - Check `agent.log` for detailed execution traces and debugging

## 🔐 Security Best Practices

1. **Never commit API keys** - Keep your GEMINI_API_KEY in environment variables, not in code
2. **Validate user input** - When building on top of this system, sanitize all user inputs
3. **Data privacy** - Student data contains PII; ensure proper access controls in production
4. **Regular backups** - Back up the `data/` directory regularly

## 🚀 Advanced Usage

### Custom Agent Integration

```python
class MyCustomAgent:
    def __init__(self):
        self.data = {}
    
    def process(self, query):
        # Your custom logic here
        return {"result": "custom response"}

# Integrate with coordinator
custom_agent = MyCustomAgent()
# Add custom routing logic in coordinator
```

### Webhook Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/query', methods=['POST'])
def process_query():
    data = request.json
    student_id = data['student_id']
    query = data['query']
    
    result = coordinator.process_query(student_id, query)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
```

### Batch Processing

```python
import csv

# Process multiple students from CSV
with open('students.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        memory_bank.add_student(row['id'], {
            'name': row['name'],
            'year': row['year'],
            'major': row['major']
        })
        
        # Generate recommendations for each
        result = coordinator.process_query(
            row['id'],
            'Generate a personalized study plan'
        )
        
        # Save results
        with open(f"results/{row['id']}.json", 'w') as out:
            json.dump(result, out, indent=2)
```

## 📞 Support

For issues, questions, or contributions:
- Check the [README.md](README.md) for project overview
- Submit issues on GitHub
- Contact: [herokugourav@gmail.com]

---

**Happy Career Preparation! 🎓**
