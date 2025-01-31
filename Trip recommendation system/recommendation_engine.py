import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler

class HybridRecommender:
    def __init__(self):
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        self.items = pd.read_csv('items.csv')
        self.ratings = pd.read_csv('ratings.csv')
        self.users = pd.read_csv('users.csv')
        
    def preprocess_data(self):
        # Collaborative Filtering Matrix
        self.user_item_matrix = self.ratings.pivot_table(
            index='userId', columns='itemId', values='rating'
        ).fillna(0)
        self.matrix = csr_matrix(self.user_item_matrix.values)
        
        # Enhanced Content-Based Features
        self.items['features'] = self.items['category'] + ' ' + \
                                self.items.get('tags', '') + ' ' + \
                                self.items.get('description', '')
        
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.items['features'])
        
        # Weighted Ratings Calculation
        C = self.items['p_rating'].mean()
        m = self.items['count'].quantile(0.85)
        self.items['weighted_rating'] = self.items.apply(
            lambda x: (x['count']/(x['count']+m) * x['p_rating']) + (m/(m+x['count']) * C), axis=1
        )

    def collaborative_recommendations(self, user_id, num_rec=5):
        try:
            model = NearestNeighbors(metric='cosine', algorithm='brute')
            model.fit(self.matrix)
            distances, indices = model.kneighbors(
                self.user_item_matrix.loc[user_id].values.reshape(1, -1),
                n_neighbors=6
            )
            similar_users = indices.flatten()[1:]
            rec_items = self.user_item_matrix.iloc[similar_users].mean(0).sort_values(ascending=False)
            seen_items = self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index
            return rec_items.drop(seen_items).head(num_rec)
        except KeyError:
            return pd.Series()

    def content_based_recommendations(self, query, num_rec=5):
        query_vec = self.tfidf.transform([query])
        sim_scores = linear_kernel(query_vec, self.tfidf_matrix).flatten()
        sim_indices = sim_scores.argsort()[::-1][:num_rec*2]
        return self.items.iloc[sim_indices], sim_scores[sim_indices]
    
    def hybrid_recommendations(self, user_id, query=None, num_rec=5):
        is_new_user = user_id not in self.user_item_matrix.index
        
        if is_new_user:
            if not query:
                return self.items.sort_values('weighted_rating', ascending=False).head(num_rec)
            
            content_rec, content_scores = self.content_based_recommendations(query, num_rec*2)
            if content_rec.empty:
                return self.items.sort_values('weighted_rating', ascending=False).head(num_rec)
            
            content_rec = content_rec.copy()
            content_rec['combined_score'] = 0.7*content_scores + 0.3*content_rec['weighted_rating']
            return content_rec.sort_values('combined_score', ascending=False).head(num_rec)
        
        else:
            # Collaborative Filtering Results
            collab_rec = self.collaborative_recommendations(user_id, num_rec)
            seen_items = self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index
            
            # Content-Based Results if query exists
            content_rec, content_scores = (self.content_based_recommendations(query, num_rec) 
                                         if query else (pd.DataFrame(), []))
            
            # Combine recommendations
            combined_items = pd.concat([
                collab_rec.rename('score'),
                pd.Series(content_rec['itemId'].tolist(), name='itemId')
            ], ignore_index=True).drop_duplicates().tolist()
            
            hybrid_df = self.items[self.items['itemId'].isin(combined_items)]
            hybrid_df = hybrid_df[~hybrid_df['itemId'].isin(seen_items)]
            
            # Merge scores
            hybrid_df = hybrid_df.merge(
                collab_rec.rename('collab_score'),
                left_on='itemId',
                right_index=True,
                how='left'
            )
            
            if query and not content_rec.empty:
                content_scores_df = pd.DataFrame({
                    'itemId': content_rec['itemId'],
                    'content_score': content_scores
                })
                hybrid_df = hybrid_df.merge(content_scores_df, on='itemId', how='left')
                hybrid_df['content_score'] = hybrid_df['content_score'].fillna(0)
            else:
                hybrid_df['content_score'] = 0
            
            # Normalize scores
            scaler = MinMaxScaler()
            score_columns = ['collab_score', 'content_score', 'weighted_rating']
            hybrid_df[score_columns] = scaler.fit_transform(hybrid_df[score_columns].fillna(0))
            
            # Weighted combination
            hybrid_df['final_score'] = (0.5 * hybrid_df['collab_score'] + 
                                      0.3 * hybrid_df['content_score'] + 
                                      0.2 * hybrid_df['weighted_rating'])
            
            return hybrid_df.sort_values('final_score', ascending=False).head(num_rec)