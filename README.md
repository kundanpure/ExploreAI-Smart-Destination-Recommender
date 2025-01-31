# Tourist Destination Recommendation System 🌴🗺️

![System Demo](https://via.placeholder.com/800x400.png?text=Hybrid+Recommendation+System+Interface)

A sophisticated hybrid recommendation system combining collaborative filtering and content-based approaches to suggest tourist destinations. Features both algorithmic implementations and a web interface for user interaction.

## Table of Contents
- [Features](#features-)
- [Installation](#installation-)
- [Usage](#usage-)
- [Data Preparation](#data-preparation-)
- [API Endpoints](#api-endpoints-)
- [Recommendation Methods](#recommendation-methods-)
- [Project Structure](#project-structure-)
- [License](#license-)
- [Contributing](#contributing-)
- [Contact](#contact-)

## Features ✨
- **Hybrid Recommendation Engine**
  - Collaborative Filtering (User-Item Matrix)
  - Content-Based Filtering (TF-IDF Vectorization)
  - Weighted Score Combination
- Adaptive User Handling
  - Personalized recommendations for existing users
  - Interest-based suggestions for new users
- Web Interface with:
  - User type detection (New/Existing)
  - Dynamic form validation
  - Responsive card-based display
  - Error handling and feedback
- Statistical Weighting System (IMDB-inspired formula)

## Technologies Used 🛠️
- **Backend**: Python 3, Flask
- **Machine Learning**: Scikit-Learn, SciPy, NumPy
- **Data Handling**: Pandas
- **Frontend**: HTML5, CSS3, JavaScript
- **Vectorization**: TF-IDF with n-gram support

## Installation 📦

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/tourist-recommendation-system.git
cd tourist-recommendation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


Sample requirements.txt:

Copy
flask==2.0.1
pandas==1.3.3
numpy==1.21.2
scikit-learn==1.0
scipy==1.7.1
jinja2==3.0.1
Usage 🚀
Running the Application
bash
Copy
flask run --host=0.0.0.0 --port=5000
Visit http://localhost:5000 in your web browser

Web Interface Guide
User Type Selection

🆕 New User: No ID required, uses interest keywords

🆔 Existing User: Provide User ID for personalized recommendations

Interest Input

Enter keywords (e.g., "beaches", "historical museums")

Supports natural language queries

Recommendation Display

Results shown in priority order

Includes ratings, review counts, and tags

Responsive cards with hover effects

Data Preparation 📂
Required CSV Files
Place these in project root:

items.csv (Destinations)

csv
Copy
itemId,title,category,tags,description,p_rating,count
1,Eiffel Tower,Landmark,"paris,architecture",A wrought-iron lattice tower...,4.8,1500
ratings.csv (User Ratings)

csv
Copy
userId,itemId,rating
1,1,5
1,2,4
users.csv (User Database)

csv
Copy
userId,name,email
1,John Doe,john@example.com
API Endpoints 🔌
Get Recommendations (POST)
POST /recommend

Parameters:

json
Copy
{
  "user_type": "existing|new",
  "user_id": 123,          // Required for existing users
  "interests": "keywords",
  "num_rec": 5             // Optional (default:5)
}
Example Response:

json
Copy
{
  "recommendations": [
    {
      "title": "Grand Canyon",
      "category": "Natural Wonder",
      "p_rating": 4.9,
      "count": 2845,
      "tags": "hiking,views",
      "score": 0.92
    }
  ]
}
Recommendation Methods 📊
Algorithmic Architecture
mermaid
Copy
graph TD
    A[User Input] --> B{User Type?}
    B -->|Existing| C[Collaborative Filtering]
    B -->|New| D[Content Filtering]
    C --> E[Hybrid Scoring]
    D --> E
    E --> F[Final Recommendations]
Technical Components
Collaborative Filtering

User-Item Matrix Factorization

KNN (k=6) with Cosine Similarity

Sparsity Handling with CSR Matrix

Content-Based Filtering

TF-IDF Vectorization (1-2 ngrams)

Feature Combination: Category + Tags + Description

Cosine Similarity Scoring

Hybrid Combination

python
Copy
final_score = (0.5 * collaborative_score) + 
             (0.3 * content_score) + 
             (0.2 * weighted_rating)
Weighted Rating Formula

python
Copy
weighted_rating = (v/(v+m) * R) + (m/(m+v) * C)
Where:

v = number of ratings

m = minimum votes required

R = average rating

C = mean rating across all items

Project Structure 📁
Copy
.
├── app.py                 # Flask application
├── recommender.py         # HybridRecommender class
├── templates/             # Web interface
│   └── index.html         # Main template
├── static/                # CSS/JS assets
├── items.csv              # Destination database
├── ratings.csv            # User ratings
├── users.csv              # User database
├── requirements.txt       # Dependencies
└── README.md              # Documentation
License 📄
This project is licensed under the MIT License - see the LICENSE file for details.

Contributing 🤝
Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add some AmazingFeature')

Push to branch (git push origin feature/AmazingFeature)

Open a Pull Request

Acknowledgements
Scikit-Learn community for ML components

Flask development team

IMDB weighted rating formula inspiration

Contact 📧
For questions or support: your.email@example.com

Happy Travel Planning! ✈️🌎

Copy

This README includes:
1. Comprehensive installation and usage instructions
2. Detailed API documentation
3. Algorithm visualization (mermaid diagram)
4. Complete file structure breakdown
5. Data schema requirements
6. Mathematical formulas and scoring logic
7. Contribution guidelines
8. License information
9. Responsive web interface details
10. Troubleshooting and support information

To use this README:
1. Replace placeholder values (yourusername, your.email@example.com)
2. Add actual screenshot URLs
3. Ensure CSV files match the specified schema
4. Create a LICENSE file
5. Verify all dependencies match your environment
