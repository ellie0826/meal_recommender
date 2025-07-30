"""
Data Preprocessing Pipeline for Meal Recommender System

This module provides comprehensive data preprocessing functionality for recipe and review data
to prepare it for a healthy meal recommendation system.
"""

import pandas as pd
import numpy as np
import re
import ast
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
from collections import Counter
from textblob import TextBlob
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeDataPreprocessor:
    """
    Preprocessor for recipe data with focus on healthy meal recommendations
    """
    
    def __init__(self):
        # Convert to sets for O(1) lookup performance
        self.healthy_keywords = {
            'low-fat', 'low-sodium', 'low-calorie', 'healthy', 'diet', 'light',
            'fresh', 'organic', 'whole-grain', 'lean', 'grilled', 'baked',
            'steamed', 'raw', 'vegetarian', 'vegan', 'gluten-free', 'sugar-free'
        }
        
        self.unhealthy_keywords = {
            'fried', 'deep-fried', 'butter', 'cream', 'heavy-cream', 'bacon',
            'sausage', 'processed', 'fast-food', 'junk', 'candy', 'dessert'
        }
        
        self.nutrition_columns = [
            'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
            'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
        ]
    
    def load_data(self, recipes_path: str, reviews_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load recipe and review data"""
        logger.info("Loading recipe and review data...")
        
        try:
            recipes = pd.read_csv(recipes_path)
            reviews = pd.read_csv(reviews_path)
            
            logger.info("Loaded %s recipes and %s reviews", len(recipes), len(reviews))
            return recipes, reviews
            
        except Exception as e:
            logger.error("Error loading data: %s", e)
            raise
    
    def clean_basic_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean basic recipe fields"""
        logger.info("Cleaning basic fields...")
        
        df = df.copy()
        
        # Remove duplicates based on RecipeId
        df = df.drop_duplicates(subset=['RecipeId'], keep='first')
        
        # Clean recipe names
        df['Name'] = df['Name'].str.strip()
        df['Name'] = df['Name'].str.title()
        
        # Clean and standardize author names
        df['AuthorName'] = df['AuthorName'].fillna('Unknown')
        df['AuthorName'] = df['AuthorName'].str.strip()
        
        # Clean descriptions
        df['Description'] = df['Description'].fillna('')
        df['Description'] = df['Description'].str.strip()
        
        return df
    
    def parse_time_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and clean time-related fields"""
        logger.info("Parsing time fields...")
        
        df = df.copy()
        
        def parse_duration(duration_str):
            """Parse ISO 8601 duration format (PT30M, PT1H30M, etc.)"""
            if pd.isna(duration_str) or duration_str == '':
                return np.nan
            
            try:
                # Remove PT prefix
                duration_str = str(duration_str).replace('PT', '')
                
                hours = 0
                minutes = 0
                
                # Extract hours
                if 'H' in duration_str:
                    hours_match = re.search(r'(\d+)H', duration_str)
                    if hours_match:
                        hours = int(hours_match.group(1))
                
                # Extract minutes
                if 'M' in duration_str:
                    minutes_match = re.search(r'(\d+)M', duration_str)
                    if minutes_match:
                        minutes = int(minutes_match.group(1))
                
                return hours * 60 + minutes  # Return total minutes
                
            except:
                return np.nan
        
        # Parse time fields
        df['CookTimeMinutes'] = df['CookTime'].apply(parse_duration)
        df['PrepTimeMinutes'] = df['PrepTime'].apply(parse_duration)
        df['TotalTimeMinutes'] = df['TotalTime'].apply(parse_duration)
        
        # Create time categories
        df['TimeCategory'] = pd.cut(df['TotalTimeMinutes'], 
                                   bins=[0, 30, 60, 120, float('inf')],
                                   labels=['Quick (≤30min)', 'Medium (30-60min)', 
                                          'Long (1-2hrs)', 'Very Long (>2hrs)'])
        
        # Parse publication date
        df['DatePublished'] = pd.to_datetime(df['DatePublished'], errors='coerce')
        df['PublicationYear'] = df['DatePublished'].dt.year
        
        return df
    
    def clean_nutrition_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process nutrition information"""
        logger.info("Cleaning nutrition data...")
        
        df = df.copy()
        
        # Clean nutrition columns
        for col in self.nutrition_columns:
            if col in df.columns:
                # Convert to numeric, handling any string values
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove extreme outliers (values beyond 99.9th percentile)
                if df[col].notna().sum() > 0:
                    upper_limit = df[col].quantile(0.999)
                    df.loc[df[col] > upper_limit, col] = np.nan
        
        # Noramlize calorie values to per serving
        df['Calories'] = df['Calories'].fillna(0)
        df['Calories'] = df['Calories'].apply(lambda x: max(x, 0))
            
        # Create healthiness score based on nutrition
        # df['HealthinessScore'] = self.calculate_healthiness_score(df)
        df = self.add_nutri_score_fields(df)
        
        # Categorize recipes by calorie content
        df['CalorieCategory'] = pd.cut(df['Calories'], 
                                      bins=[0, 200, 400, 600, float('inf')],
                                      labels=['Low (<200)', 'Medium (200-400)', 
                                             'High (400-600)', 'Very High (>600)'])
        
        return df

    def calculate_nutri_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the numeric Nutri-Score based on EU guidelines"""

        df = df.copy()

        def score_energy(kj):
            thresholds = [335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350]
            return sum(kj > t for t in thresholds)

        def score_sugar(g):
            thresholds = [4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45]
            return sum(g > t for t in thresholds)

        def score_sat_fat(g):
            thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            return sum(g > t for t in thresholds)

        def score_sodium(mg):
            thresholds = [90, 180, 270, 360, 450, 540, 630, 720, 810, 900]
            return sum(mg > t for t in thresholds)

        def score_fiber(g):
            thresholds = [0.9, 1.9, 2.8, 3.7, 4.7]
            return sum(g > t for t in thresholds)

        def score_protein(g):
            thresholds = [1.6, 3.2, 4.8, 6.4, 8.0]
            return sum(g > t for t in thresholds)

        # Apply scoring row by row
        def compute_row_score(row):
            energy_kj = row.get('Calories', 0) * 4.184
            sugar = row.get('SugarContent', 0)
            sat_fat = row.get('SaturatedFatContent', 0)
            sodium = row.get('SodiumContent', 0)
            fiber = row.get('FiberContent', 0)
            protein = row.get('ProteinContent', 0)

            neg = (
                score_energy(energy_kj) +
                score_sugar(sugar) +
                score_sat_fat(sat_fat) +
                score_sodium(sodium)
            )
            pos = score_fiber(fiber) + score_protein(protein)

            return neg - pos

        return df.apply(compute_row_score, axis=1)


    def map_nutri_score_grade(self, score: float) -> str:
        """Map numeric Nutri-Score to letter grade"""
        if score <= 2:
            return 'A'
        elif score <= 10:
            return 'B'
        elif score <= 18:
            return 'C'
        elif score <= 26:
            return 'D'
        else:
            return 'E'


    def add_nutri_score_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Nutri-Score value, grade, and binary health flag to dataframe"""
        logger.info("Adding Nutri-Score fields...")

        df = df.copy()

        df['NutriScore'] = self.calculate_nutri_score(df)
        df['NutriScoreGrade'] = df['NutriScore'].apply(self.map_nutri_score_grade)
        df['IsNutriHealthy'] = df['NutriScoreGrade'].isin(['A', 'B'])

        return df

    def normalize_nutrition_per_serving(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize nutrition values to per serving basis"""
        logger.info("Normalizing nutrition values per serving...")
        
        df = df.copy()
        
        # Clean and process serving size information
        df['RecipeServings'] = pd.to_numeric(df['RecipeServings'], errors='coerce')
        
        # Handle extreme serving sizes (likely data errors)
        # Cap at reasonable maximum (e.g., 50 servings) and minimum (1 serving)
        df['RecipeServings'] = df['RecipeServings'].clip(lower=1, upper=50)
        
        # Fill missing serving sizes with median (around 6 servings based on data exploration)
        median_servings = df['RecipeServings'].median()
        df['RecipeServings'] = df['RecipeServings'].fillna(median_servings)
        
        # Create normalized nutrition columns (per serving)
        nutrition_cols_to_normalize = [
            'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
            'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
        ]
        
        for col in nutrition_cols_to_normalize:
            if col in df.columns:
                # Create per-serving column
                per_serving_col = f'{col}PerServing'
                df[per_serving_col] = df[col] / df['RecipeServings']
                
                # Replace original column with per-serving values for consistency
                df[col] = df[per_serving_col]
        
        # Add serving size category for analysis
        df['ServingSizeCategory'] = pd.cut(df['RecipeServings'], 
                                         bins=[0, 2, 4, 8, float('inf')],
                                         labels=['Individual (1-2)', 'Small (3-4)', 
                                                'Medium (5-8)', 'Large (9+)'])
        
        return df


    def process_ingredients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process ingredient information"""
        logger.info("Processing ingredients...")
        
        df = df.copy()
        
        def safe_eval(x):
            """Safely evaluate string representations of lists"""
            if pd.isna(x) or x == '':
                return []
            try:
                if isinstance(x, str):
                    # Clean the string and evaluate
                    x = x.replace('\n', ' ').strip()
                    return ast.literal_eval(x)
                return x if isinstance(x, list) else []
            except (ValueError, SyntaxError):
                return []
        
        # Parse ingredient lists
        df['IngredientList'] = df['RecipeIngredientParts'].apply(safe_eval)
        df['IngredientQuantities'] = df['RecipeIngredientQuantities'].apply(safe_eval)
        
        # Count ingredients
        df['IngredientCount'] = df['IngredientList'].apply(len)
        
        # Categorize by ingredient complexity
        df['ComplexityCategory'] = pd.cut(df['IngredientCount'], 
                                         bins=[0, 5, 10, 15, float('inf')],
                                         labels=['Simple (≤5)', 'Medium (6-10)', 
                                                'Complex (11-15)', 'Very Complex (>15)'])
        
        # Extract healthy/unhealthy ingredient indicators
        df['HasHealthyIngredients'] = df['IngredientList'].apply(
            lambda x: any(any(keyword in str(ingredient).lower() 
                             for keyword in self.healthy_keywords) 
                         for ingredient in x)
        )
        
        df['HasUnhealthyIngredients'] = df['IngredientList'].apply(
            lambda x: any(any(keyword in str(ingredient).lower() 
                             for keyword in self.unhealthy_keywords) 
                         for ingredient in x)
        )
        
        return df
    
    def process_categories_and_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process recipe categories and keywords"""
        logger.info("Processing categories and keywords...")
        
        df = df.copy()
        
        # Clean categories
        df['RecipeCategory'] = df['RecipeCategory'].fillna('Other')
        df['RecipeCategory'] = df['RecipeCategory'].str.strip()
        
        # Process keywords
        def safe_eval_keywords(x):
            if pd.isna(x) or x == '':
                return []
            try:
                if isinstance(x, str):
                    x = x.replace('\n', ' ').strip()
                    return ast.literal_eval(x)
                return x if isinstance(x, list) else []
            except:
                return []
        
        df['KeywordList'] = df['Keywords'].apply(safe_eval_keywords)
        
        # Create dietary restriction flags
        df['IsVegetarian'] = df['KeywordList'].apply(
            lambda x: any('vegetarian' in str(keyword).lower() for keyword in x)
        )
        
        df['IsVegan'] = df['KeywordList'].apply(
            lambda x: any('vegan' in str(keyword).lower() for keyword in x)
        )
        
        df['IsGlutenFree'] = df['KeywordList'].apply(
            lambda x: any('gluten' in str(keyword).lower() and 'free' in str(keyword).lower() 
                         for keyword in x)
        )
        
        df['IsLowCarb'] = df['KeywordList'].apply(
            lambda x: any('low' in str(keyword).lower() and 'carb' in str(keyword).lower() 
                         for keyword in x)
        )

        # Pre-compile regex patterns for optimal performance
        healthy_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in self.healthy_keywords), 
            re.IGNORECASE
        )
        unhealthy_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in self.unhealthy_keywords), 
            re.IGNORECASE
        )
        
        def check_keywords_optimized(keyword_list, pattern):
            """Optimized keyword checking with compiled regex"""
            if not keyword_list:
                return False
            # Join all keywords into a single string for one regex search
            text = ' '.join(str(kw) for kw in keyword_list)
            return bool(pattern.search(text))
        
        df['HasHealthyKeywords'] = df['KeywordList'].apply(
            lambda x: check_keywords_optimized(x, healthy_pattern)
        )
        
        df['HasUnhealthyKeywords'] = df['KeywordList'].apply(
            lambda x: check_keywords_optimized(x, unhealthy_pattern)
        )

        df['HasAnyHealthyTag'] = df['HasHealthyIngredients'] | df['HasHealthyKeywords']
        df['HasAnyUnhealthyTag'] = df['HasUnhealthyIngredients'] | df['HasUnhealthyKeywords']
        df['FinalHealthinessFlag'] = (
            df['IsNutriHealthy'] & ~df['HasAnyUnhealthyTag']
        )

        
        return df
    
    def process_ratings_and_reviews(self, recipes_df: pd.DataFrame, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Process rating and review information"""
        logger.info("Processing ratings and reviews...")
        
        # Aggregate review statistics
        review_stats = reviews_df.groupby('RecipeId').agg({
            'Rating': ['mean', 'count', 'std'],
            'Review': lambda x: x.str.len().mean()  # Average review length
        }).round(2)
        
        # Flatten column names
        review_stats.columns = ['AvgRating', 'ReviewCount', 'RatingStd', 'AvgReviewLength']
        review_stats = review_stats.reset_index()
        
        # Merge with recipes
        recipes_enhanced = recipes_df.merge(review_stats, on='RecipeId', how='left')
        
        # Handle column name conflicts from merge
        # The original dataset has ReviewCount, so after merge we get ReviewCount_x (original) and ReviewCount_y (from reviews)
        if 'ReviewCount_y' in recipes_enhanced.columns:
            # Use the calculated review count from actual reviews
            recipes_enhanced['ReviewCount'] = recipes_enhanced['ReviewCount_y']
            # Drop the conflicting columns
            recipes_enhanced = recipes_enhanced.drop(['ReviewCount_x', 'ReviewCount_y'], axis=1)
        
        # Ensure columns exist and fill missing values
        if 'AvgRating' not in recipes_enhanced.columns:
            recipes_enhanced['AvgRating'] = np.nan
        if 'ReviewCount' not in recipes_enhanced.columns:
            recipes_enhanced['ReviewCount'] = 0
        if 'RatingStd' not in recipes_enhanced.columns:
            recipes_enhanced['RatingStd'] = 0
        if 'AvgReviewLength' not in recipes_enhanced.columns:
            recipes_enhanced['AvgReviewLength'] = 0
            
        # Fill missing values
        if 'AggregatedRating' in recipes_enhanced.columns:
            recipes_enhanced['AvgRating'] = recipes_enhanced['AvgRating'].fillna(
                recipes_enhanced['AggregatedRating']
            )
        else:
            recipes_enhanced['AvgRating'] = recipes_enhanced['AvgRating'].fillna(0)
            
        recipes_enhanced['ReviewCount'] = recipes_enhanced['ReviewCount'].fillna(0)
        recipes_enhanced['RatingStd'] = recipes_enhanced['RatingStd'].fillna(0)
        recipes_enhanced['AvgReviewLength'] = recipes_enhanced['AvgReviewLength'].fillna(0)
        
        # Create popularity score
        recipes_enhanced['PopularityScore'] = (
            recipes_enhanced['AvgRating'] * 0.7 + 
            np.log1p(recipes_enhanced['ReviewCount']) * 0.3
        )
        
        return recipes_enhanced
    
    def create_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create final engineered features for the recommender system"""
        logger.info("Creating final features...")
        
        df = df.copy()
        
        # Create composite recommendation score
        df['RecommendationScore'] = (
            (100 - df['NutriScore']) * 0.4 +
            df['PopularityScore'] * 10 * 0.3 +  # Scale popularity score
            (100 - df['TotalTimeMinutes'].fillna(60) / 2) * 0.2 +  # Time efficiency bonus
            df['FinalHealthinessFlag'].astype(int) * 10 * 0.1
        )
        
        # Normalize recommendation score to 0-100
        df['RecommendationScore'] = np.clip(df['RecommendationScore'], 0, 100)
        
        # Create meal type categories based on time and ingredients
        def categorize_meal_type(row):
            if row['TotalTimeMinutes'] <= 15:
                return 'Quick Snack'
            elif row['TotalTimeMinutes'] <= 30:
                return 'Light Meal'
            elif row['TotalTimeMinutes'] <= 60:
                return 'Regular Meal'
            else:
                return 'Special Occasion'
        
        df['MealType'] = df.apply(categorize_meal_type, axis=1)
        
        return df
    
    def filter_for_healthy_recommendations(self, df: pd.DataFrame,
                                         min_rating: float = 3.5,
                                         max_calories: float = 800) -> pd.DataFrame:
        """Filter recipes suitable for healthy meal recommendations"""
        logger.info("Filtering for healthy recommendations...")
        
        filtered_df = df[
            (df['FinalHealthinessFlag']) &
            (df['AvgRating'] >= min_rating) &
            (df['Calories'] <= max_calories) &
            (df['ReviewCount'] >= 5)  # Ensure sufficient reviews
        ].copy()
        
        logger.info("Filtered to %s healthy recipes from %s total recipes", len(filtered_df), len(df))
        
        return filtered_df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data"""
        logger.info("Saving processed data to %s", output_path)
        
        # Select key columns for the recommender system
        key_columns = [
            'RecipeId', 'Name', 'Description', 'RecipeCategory',
            'TotalTimeMinutes', 'TimeCategory', 'IngredientCount', 'ComplexityCategory',
            'Calories', 'CalorieCategory', 'NutriScore', 'NutriScoreGrade', 'IsNutriHealthy', 
            'RecommendationScore', 'AvgRating', 'ReviewCount',
            'PopularityScore','IsVegetarian', 'IsVegan', 'IsGlutenFree', 
            'IsLowCarb', 'HasHealthyIngredients', 'HasUnhealthyIngredients',
            'MealType', 'IngredientList', 'RecipeServings', 'ServingSizeCategory'
        ] + [col for col in self.nutrition_columns if col in df.columns]
        
        # Select only available columns
        available_columns = [col for col in key_columns if col in df.columns]
        
        df[available_columns].to_csv(output_path, index=False)
        logger.info("Data saved successfully")


class ReviewPreprocessor:
    """
    Preprocessor for user review data to support personalized meal recommendations
    using two-towers model architecture
    """
    
    def __init__(self):
        # Health-related keywords for review analysis
        self.health_positive_keywords = {
            'healthy', 'nutritious', 'fresh', 'light', 'clean', 'wholesome',
            'balanced', 'lean', 'low-fat', 'low-calorie', 'diet-friendly',
            'guilt-free', 'energizing', 'satisfying', 'nourishing'
        }
        
        self.health_negative_keywords = {
            'heavy', 'greasy', 'oily', 'fatty', 'unhealthy', 'junk',
            'processed', 'artificial', 'sugary', 'salty', 'fried',
            'caloric', 'indulgent', 'rich', 'decadent'
        }
        
        # Taste and quality keywords
        self.taste_positive_keywords = {
            'delicious', 'tasty', 'flavorful', 'amazing', 'excellent', 'perfect',
            'wonderful', 'fantastic', 'great', 'good', 'love', 'favorite',
            'yummy', 'scrumptious', 'divine', 'incredible', 'outstanding'
        }
        
        self.taste_negative_keywords = {
            'bland', 'tasteless', 'awful', 'terrible', 'disgusting', 'horrible',
            'bad', 'worst', 'hate', 'nasty', 'gross', 'disappointing',
            'flavorless', 'boring', 'mediocre', 'poor'
        }
        
        # Difficulty and preparation keywords
        self.easy_keywords = {
            'easy', 'simple', 'quick', 'fast', 'effortless', 'straightforward',
            'basic', 'beginner', 'no-fuss', 'hassle-free'
        }
        
        self.difficult_keywords = {
            'difficult', 'hard', 'complex', 'complicated', 'challenging',
            'time-consuming', 'tedious', 'advanced', 'tricky'
        }
    
    def load_review_data(self, reviews_path: str) -> pd.DataFrame:
        """Load review data"""
        logger.info("Loading review data...")
        
        try:
            reviews = pd.read_csv(reviews_path)
            logger.info("Loaded %s reviews", len(reviews))
            return reviews
        except Exception as e:
            logger.error("Error loading review data: %s", e)
            raise
    
    def clean_review_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess review text"""
        logger.info("Cleaning review text...")
        
        df = df.copy()
        
        # Handle missing reviews
        df['Review'] = df['Review'].fillna('')
        
        # Convert to string and clean
        df['Review'] = df['Review'].astype(str)
        
        # Remove extra whitespace and normalize
        df['Review'] = df['Review'].str.strip()
        df['Review'] = df['Review'].str.replace(r'\s+', ' ', regex=True)
        
        # Convert to lowercase for processing (keep original for display)
        df['ReviewLower'] = df['Review'].str.lower()
        
        # Remove punctuation for keyword analysis
        df['ReviewClean'] = df['ReviewLower'].str.translate(
            str.maketrans('', '', string.punctuation)
        )
        
        # Calculate review length metrics
        df['ReviewLength'] = df['Review'].str.len()
        df['ReviewWordCount'] = df['ReviewClean'].str.split().str.len()
        
        # Filter out very short reviews (likely not informative)
        df['IsSubstantialReview'] = df['ReviewWordCount'] >= 3
        
        return df
    
    def extract_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract sentiment features from reviews using TextBlob"""
        logger.info("Extracting sentiment features...")
        
        df = df.copy()
        
        def get_sentiment(text):
            """Get sentiment polarity and subjectivity"""
            if not text or len(text.strip()) == 0:
                return 0.0, 0.0
            try:
                blob = TextBlob(text)
                return blob.sentiment.polarity, blob.sentiment.subjectivity
            except:
                return 0.0, 0.0
        
        # Apply sentiment analysis
        sentiment_results = df['Review'].apply(get_sentiment)
        df['SentimentPolarity'] = sentiment_results.apply(lambda x: x[0])
        df['SentimentSubjectivity'] = sentiment_results.apply(lambda x: x[1])
        
        # Categorize sentiment
        df['SentimentCategory'] = pd.cut(
            df['SentimentPolarity'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        # Create binary sentiment flags
        df['IsPositiveSentiment'] = df['SentimentPolarity'] > 0.1
        df['IsNegativeSentiment'] = df['SentimentPolarity'] < -0.1
        
        return df
    
    def extract_keyword_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract keyword-based features from reviews"""
        logger.info("Extracting keyword features...")
        
        df = df.copy()
        
        # Compile regex patterns for efficiency
        health_pos_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in self.health_positive_keywords),
            re.IGNORECASE
        )
        health_neg_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in self.health_negative_keywords),
            re.IGNORECASE
        )
        taste_pos_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in self.taste_positive_keywords),
            re.IGNORECASE
        )
        taste_neg_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in self.taste_negative_keywords),
            re.IGNORECASE
        )
        easy_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in self.easy_keywords),
            re.IGNORECASE
        )
        difficult_pattern = re.compile(
            '|'.join(re.escape(kw) for kw in self.difficult_keywords),
            re.IGNORECASE
        )
        
        # Extract keyword features
        df['HasHealthPositive'] = df['Review'].str.contains(health_pos_pattern, na=False)
        df['HasHealthNegative'] = df['Review'].str.contains(health_neg_pattern, na=False)
        df['HasTastePositive'] = df['Review'].str.contains(taste_pos_pattern, na=False)
        df['HasTasteNegative'] = df['Review'].str.contains(taste_neg_pattern, na=False)
        df['HasEasyKeywords'] = df['Review'].str.contains(easy_pattern, na=False)
        df['HasDifficultKeywords'] = df['Review'].str.contains(difficult_pattern, na=False)
        
        # Count keyword occurrences
        df['HealthPositiveCount'] = df['Review'].str.count(health_pos_pattern)
        df['HealthNegativeCount'] = df['Review'].str.count(health_neg_pattern)
        df['TastePositiveCount'] = df['Review'].str.count(taste_pos_pattern)
        df['TasteNegativeCount'] = df['Review'].str.count(taste_neg_pattern)
        
        # Create composite health perception score
        df['HealthPerceptionScore'] = (
            df['HealthPositiveCount'] - df['HealthNegativeCount']
        )
        
        # Create composite taste perception score
        df['TastePerceptionScore'] = (
            df['TastePositiveCount'] - df['TasteNegativeCount']
        )
        
        return df
    
    def process_user_behavior(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process user behavior patterns from reviews"""
        logger.info("Processing user behavior patterns...")
        
        df = df.copy()
        
        # Parse review date
        df['ReviewDate'] = pd.to_datetime(df['DateSubmitted'], errors='coerce')
        df['ReviewYear'] = df['ReviewDate'].dt.year
        df['ReviewMonth'] = df['ReviewDate'].dt.month
        df['ReviewDayOfWeek'] = df['ReviewDate'].dt.dayofweek
        
        # Calculate user-level statistics
        user_stats = df.groupby('AuthorId').agg({
            'Rating': ['mean', 'std', 'count'],
            'ReviewLength': 'mean',
            'SentimentPolarity': 'mean',
            'HealthPerceptionScore': 'mean',
            'TastePerceptionScore': 'mean'
        }).round(3)
        
        # Flatten column names
        user_stats.columns = [
            'UserAvgRating', 'UserRatingStd', 'UserReviewCount',
            'UserAvgReviewLength', 'UserAvgSentiment',
            'UserAvgHealthPerception', 'UserAvgTastePerception'
        ]
        user_stats = user_stats.reset_index()
        
        # Merge user statistics back to reviews
        df = df.merge(user_stats, on='AuthorId', how='left')
        
        # Create user behavior categories
        df['UserEngagementLevel'] = pd.cut(
            df['UserReviewCount'],
            bins=[0, 1, 5, 20, float('inf')],
            labels=['One-time', 'Occasional', 'Regular', 'Power User']
        )
        
        df['UserRatingBehavior'] = pd.cut(
            df['UserAvgRating'],
            bins=[0, 3, 4, 4.5, 5],
            labels=['Critical', 'Moderate', 'Positive', 'Very Positive']
        )
        
        # Identify health-conscious users
        df['IsHealthConsciousUser'] = df['UserAvgHealthPerception'] > 0.5
        
        return df
    
    def create_user_recipe_features(self, df: pd.DataFrame, recipes_df: pd.DataFrame) -> pd.DataFrame:
        """Create features linking users to recipe characteristics"""
        logger.info("Creating user-recipe interaction features...")
        
        df = df.copy()
        
        # Merge with recipe data to get recipe characteristics
        recipe_features = recipes_df[[
            'RecipeId', 'NutriScore', 'NutriScoreGrade', 'IsNutriHealthy',
            'Calories', 'TotalTimeMinutes', 'IngredientCount',
            'IsVegetarian', 'IsVegan', 'IsGlutenFree', 'IsLowCarb',
            'FinalHealthinessFlag'
        ]].copy()
        
        df = df.merge(recipe_features, on='RecipeId', how='left')
        
        # Create interaction features
        df['HealthyRecipePositiveReview'] = (
            df['FinalHealthinessFlag'] & df['IsPositiveSentiment']
        )
        
        df['HealthyRecipeNegativeReview'] = (
            df['FinalHealthinessFlag'] & df['IsNegativeSentiment']
        )
        
        df['UserHealthRecipeAlignment'] = (
            df['IsHealthConsciousUser'] & df['FinalHealthinessFlag']
        )
        
        # Calculate user preferences for recipe characteristics
        user_recipe_prefs = df.groupby('AuthorId').agg({
            'FinalHealthinessFlag': 'mean',
            'IsVegetarian': 'mean',
            'IsVegan': 'mean',
            'IsGlutenFree': 'mean',
            'IsLowCarb': 'mean',
            'Calories': 'mean',
            'TotalTimeMinutes': 'mean',
            'NutriScore': 'mean'
        }).round(3)
        
        user_recipe_prefs.columns = [
            'UserHealthyRecipePreference', 'UserVegetarianPreference',
            'UserVeganPreference', 'UserGlutenFreePreference',
            'UserLowCarbPreference', 'UserAvgCaloriePreference',
            'UserAvgTimePreference', 'UserAvgNutriScorePreference'
        ]
        user_recipe_prefs = user_recipe_prefs.reset_index()
        
        df = df.merge(user_recipe_prefs, on='AuthorId', how='left')
        
        return df
    
    def create_two_towers_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create features optimized for two-towers model architecture"""
        logger.info("Creating two-towers model features...")
        
        # User tower features
        user_features = df.groupby('AuthorId').agg({
            'UserAvgRating': 'first',
            'UserReviewCount': 'first',
            'UserAvgSentiment': 'first',
            'UserAvgHealthPerception': 'first',
            'UserAvgTastePerception': 'first',
            'UserHealthyRecipePreference': 'first',
            'UserVegetarianPreference': 'first',
            'UserVeganPreference': 'first',
            'UserGlutenFreePreference': 'first',
            'UserLowCarbPreference': 'first',
            'UserAvgCaloriePreference': 'first',
            'UserAvgTimePreference': 'first',
            'UserAvgNutriScorePreference': 'first',
            'IsHealthConsciousUser': 'first',
            'UserEngagementLevel': 'first',
            'UserRatingBehavior': 'first'
        }).reset_index()
        
        # Recipe tower features (aggregated from reviews)
        recipe_features = df.groupby('RecipeId').agg({
            'Rating': ['mean', 'count', 'std'],
            'SentimentPolarity': 'mean',
            'HealthPerceptionScore': 'mean',
            'TastePerceptionScore': 'mean',
            'ReviewLength': 'mean',
            'HealthyRecipePositiveReview': 'sum',
            'HealthyRecipeNegativeReview': 'sum',
            'NutriScore': 'first',
            'IsNutriHealthy': 'first',
            'FinalHealthinessFlag': 'first',
            'Calories': 'first',
            'TotalTimeMinutes': 'first'
        }).round(3)
        
        # Flatten recipe feature column names
        recipe_features.columns = [
            'RecipeAvgRating', 'RecipeReviewCount', 'RecipeRatingStd',
            'RecipeAvgSentiment', 'RecipeAvgHealthPerception',
            'RecipeAvgTastePerception', 'RecipeAvgReviewLength',
            'RecipeHealthyPositiveCount', 'RecipeHealthyNegativeCount',
            'RecipeNutriScore', 'RecipeIsNutriHealthy',
            'RecipeFinalHealthinessFlag', 'RecipeCalories', 'RecipeTotalTime'
        ]
        recipe_features = recipe_features.reset_index()
        
        # Create recipe health perception ratio
        recipe_features['RecipeHealthPerceptionRatio'] = (
            recipe_features['RecipeHealthyPositiveCount'] / 
            (recipe_features['RecipeHealthyPositiveCount'] + 
             recipe_features['RecipeHealthyNegativeCount'] + 1)  # Add 1 to avoid division by zero
        )
        
        logger.info("Created user features: %s users", len(user_features))
        logger.info("Created recipe features: %s recipes", len(recipe_features))
        
        return user_features, recipe_features
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed review data"""
        logger.info("Saving processed review data to %s", output_path)
        
        # Select key columns for the recommender system
        key_columns = [
            'AuthorId', 'RecipeId', 'Rating', 'Review', 'ReviewDate',
            'ReviewLength', 'ReviewWordCount', 'IsSubstantialReview',
            'SentimentPolarity', 'SentimentSubjectivity', 'SentimentCategory',
            'IsPositiveSentiment', 'IsNegativeSentiment',
            'HasHealthPositive', 'HasHealthNegative', 'HasTastePositive', 'HasTasteNegative',
            'HealthPerceptionScore', 'TastePerceptionScore',
            'UserAvgRating', 'UserReviewCount', 'UserAvgSentiment',
            'UserHealthyRecipePreference', 'IsHealthConsciousUser',
            'UserEngagementLevel', 'UserRatingBehavior',
            'HealthyRecipePositiveReview', 'UserHealthRecipeAlignment'
        ]
        
        # Select only available columns
        available_columns = [col for col in key_columns if col in df.columns]
        
        df[available_columns].to_csv(output_path, index=False)
        logger.info("Processed review data saved successfully")
    
    def save_two_towers_features(self, user_features: pd.DataFrame, 
                                recipe_features: pd.DataFrame, 
                                user_output_path: str, 
                                recipe_output_path: str):
        """Save two-towers model features"""
        logger.info("Saving two-towers features...")
        
        user_features.to_csv(user_output_path, index=False)
        recipe_features.to_csv(recipe_output_path, index=False)
        
        logger.info("Two-towers features saved successfully")
        logger.info("User features saved to: %s", user_output_path)
        logger.info("Recipe features saved to: %s", recipe_output_path)

# def main():
#     """Main preprocessing pipeline"""
#     # Initialize preprocessor
#     preprocessor = RecipeDataPreprocessor()
    
#     # Load data
#     recipes_df, reviews_df = preprocessor.load_data(
#         'data/recipes.csv', 
#         'data/reviews.csv'
#     )
    
#     # Apply preprocessing steps
#     recipes_df = preprocessor.clean_basic_fields(recipes_df)
#     recipes_df = preprocessor.parse_time_fields(recipes_df)
#     recipes_df = preprocessor.normalize_nutrition_per_serving(recipes_df)
#     recipes_df = preprocessor.clean_nutrition_data(recipes_df)
#     recipes_df = preprocessor.process_ingredients(recipes_df)
#     recipes_df = preprocessor.process_categories_and_keywords(recipes_df)
#     recipes_df = preprocessor.process_ratings_and_reviews(recipes_df, reviews_df)
#     recipes_df = preprocessor.create_final_features(recipes_df)
    
#     # Filter for healthy recommendations
#     healthy_recipes = preprocessor.filter_for_healthy_recommendations(recipes_df)
    
#     # Save processed data
#     preprocessor.save_processed_data(recipes_df, 'data/processed_recipes_all.csv')
#     preprocessor.save_processed_data(healthy_recipes, 'data/processed_recipes_healthy.csv')
    
#     print("\nPreprocessing completed successfully!")
#     print(f"Total recipes processed: {len(recipes_df)}")
#     print(f"Healthy recipes for recommendations: {len(healthy_recipes)}")


def main_review_preprocessing():
    """Main review preprocessing pipeline for two-towers model"""
    # Initialize preprocessors
    recipe_preprocessor = RecipeDataPreprocessor()
    review_preprocessor = ReviewPreprocessor()
    
    # Load data
    recipes_df, reviews_df = recipe_preprocessor.load_data(
        'data/recipes.csv', 
        'data/reviews.csv'
    )
    
    # Process recipes first (needed for user-recipe features)
    print("\n=== Processing Recipes ===")
    recipes_df = recipe_preprocessor.clean_basic_fields(recipes_df)
    recipes_df = recipe_preprocessor.parse_time_fields(recipes_df)
    recipes_df = recipe_preprocessor.normalize_nutrition_per_serving(recipes_df)
    recipes_df = recipe_preprocessor.clean_nutrition_data(recipes_df)
    recipes_df = recipe_preprocessor.process_ingredients(recipes_df)
    recipes_df = recipe_preprocessor.process_categories_and_keywords(recipes_df)
    recipes_df = recipe_preprocessor.process_ratings_and_reviews(recipes_df, reviews_df)
    recipes_df = recipe_preprocessor.create_final_features(recipes_df)

    # Filter for healthy recommendations
    healthy_recipes = recipe_preprocessor.filter_for_healthy_recommendations(recipes_df)
    
    # Save processed data
    recipe_preprocessor.save_processed_data(recipes_df, 'data/processed_recipes_all.csv')
    recipe_preprocessor.save_processed_data(healthy_recipes, 'data/processed_recipes_healthy.csv')
    
    print("\nPreprocessing completed successfully!")
    print(f"Total recipes processed: {len(recipes_df)}")
    print(f"Healthy recipes for recommendations: {len(healthy_recipes)}")
    
    # Process reviews for two-towers model
    print("\n=== Processing Reviews for Two-Towers Model ===")
    reviews_df = review_preprocessor.clean_review_text(reviews_df)
    reviews_df = review_preprocessor.extract_sentiment_features(reviews_df)
    reviews_df = review_preprocessor.extract_keyword_features(reviews_df)
    reviews_df = review_preprocessor.process_user_behavior(reviews_df)
    reviews_df = review_preprocessor.create_user_recipe_features(reviews_df, recipes_df)
    
    # Create two-towers features
    print("\n=== Creating Two-Towers Features ===")
    user_features, recipe_features = review_preprocessor.create_two_towers_features(reviews_df)
    
    # Save processed data
    print("\n=== Saving Processed Data ===")
    review_preprocessor.save_processed_data(reviews_df, 'data/processed_reviews.csv')
    review_preprocessor.save_two_towers_features(
        user_features, recipe_features,
        'data/user_features_two_towers.csv',
        'data/recipe_features_two_towers.csv'
    )
    
    print("\n=== Review Preprocessing Completed Successfully! ===")
    print(f"Total reviews processed: {len(reviews_df)}")
    print(f"Unique users: {len(user_features)}")
    print(f"Unique recipes with reviews: {len(recipe_features)}")
    print(f"Health-conscious users: {reviews_df['IsHealthConsciousUser'].sum()}")
    print(f"Reviews with positive health perception: {(reviews_df['HealthPerceptionScore'] > 0).sum()}")


if __name__ == "__main__":
    main_review_preprocessing()