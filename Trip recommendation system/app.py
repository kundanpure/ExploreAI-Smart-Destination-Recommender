from flask import Flask, render_template, request
import pandas as pd
from recommendation_engine import HybridRecommender

app = Flask(__name__)
recommender = HybridRecommender()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_type = request.form['user_type']
    interests = request.form['interests']
    
    try:
        if user_type == 'existing':
            user_id = int(request.form['user_id'])
        else:
            user_id = -1  # Dummy ID for new users
        
        recommendations = recommender.hybrid_recommendations(
            user_id=user_id,
            query=interests
        )
        
        return render_template('index.html', 
                             recommendations=recommendations.to_dict('records'),
                             show_results=True)
    
    except Exception as e:
        return render_template('index.html', 
                             error_message=str(e),
                             show_results=False)

if __name__ == '__main__':
    app.run(debug=True)