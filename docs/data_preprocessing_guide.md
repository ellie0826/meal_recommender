# Data Preprocessing Guide for Meal Recommender System

## Overview

This guide outlines the comprehensive data preprocessing strategy for building a healthy meal recommender system. The preprocessing pipeline transforms raw recipe and review data into clean, feature-rich datasets optimized for recommendation algorithms.

## Data Sources

### Recipe Data (`recipes.csv`)
- **Size**: ~4.2M records
- **Key Fields**: Recipe details, nutrition information, ingredients, cooking times, categories
- **Challenges**: Complex nested data structures, missing values, inconsistent formats

### Review Data (`reviews.csv`)
- **Size**: ~1.6M records  
- **Key Fields**: User ratings, review text, timestamps
- **Purpose**: Provides popularity and quality signals for recommendations

## Preprocessing Strategy

### 1. Data Cleaning & Standardization

#### Basic Field Cleaning
- **Duplicate Removal**: Remove duplicate recipes based on `RecipeId`
- **Text Standardization**: Clean and standardize recipe names, author names, descriptions
- **Missing Value Handling**: Fill missing values with appropriate defaults

#### Time Field Processing
- **Duration Parsing**: Convert ISO 8601 duration formats (PT30M, PT1H30M) to minutes
- **Time Categorization**: Create categorical time buckets (Quick ≤30min, Medium 30-60min, etc.)
- **Date Processing**: Parse publication dates and extract year information

### 2. Nutrition Data Processing

#### Data Quality Improvements
- **Outlier Removal**: Remove extreme outliers beyond 99.9th percentile
- **Type Conversion**: Ensure all nutrition values are numeric
- **Missing Value Imputation**: Handle missing nutrition data appropriately

#### Health-Focused Feature Engineering
- **Healthiness Score**: Composite score based on nutritional content
  - Penalties for: High calories, fat, saturated fat, sodium, sugar
  - Bonuses for: High fiber, protein content
  - Normalized to 0-100 scale

- **Calorie Categories**: Low (<200), Medium (200-400), High (400-600), Very High (>600)

### 3. Ingredient Processing

#### Data Structure Handling
- **List Parsing**: Safely parse string representations of ingredient lists
- **Ingredient Counting**: Count total ingredients per recipe
- **Complexity Categorization**: Simple (≤5), Medium (6-10), Complex (11-15), Very Complex (>15)

#### Health Indicators
- **Healthy Ingredient Detection**: Flag recipes with healthy keywords
  - Keywords: low-fat, low-sodium, organic, whole-grain, lean, grilled, baked, etc.
- **Unhealthy Ingredient Detection**: Flag recipes with unhealthy keywords
  - Keywords: fried, deep-fried, processed, heavy-cream, etc.

### 4. Category & Dietary Restriction Processing

#### Recipe Categorization
- **Category Standardization**: Clean and standardize recipe categories
- **Keyword Processing**: Parse and process recipe keywords

#### Dietary Flags
- **Vegetarian Detection**: Identify vegetarian recipes
- **Vegan Detection**: Identify vegan recipes  
- **Gluten-Free Detection**: Identify gluten-free recipes
- **Low-Carb Detection**: Identify low-carb recipes

### 5. Rating & Review Integration

#### Review Aggregation
- **Average Rating**: Calculate mean rating per recipe
- **Review Count**: Count total reviews per recipe
- **Rating Variability**: Calculate standard deviation of ratings
- **Review Quality**: Average review length as quality indicator

#### Popularity Scoring
- **Composite Score**: Combine average rating (70%) with log-scaled review count (30%)
- **Missing Value Handling**: Use aggregated ratings when review data is missing

### 6. Advanced Feature Engineering

#### Recommendation Score
Composite score combining multiple factors:
- **Healthiness Score** (40%): Nutritional quality
- **Popularity Score** (30%): User ratings and engagement
- **Time Efficiency** (20%): Cooking time considerations
- **Healthy Ingredients** (10%): Presence of healthy ingredients

#### Meal Type Classification
Based on cooking time and complexity:
- **Quick Snack**: ≤15 minutes
- **Light Meal**: 15-30 minutes
- **Regular Meal**: 30-60 minutes
- **Special Occasion**: >60 minutes

### 7. Filtering for Healthy Recommendations

#### Quality Filters
- **Minimum Healthiness Score**: ≥60 (configurable)
- **Minimum Rating**: ≥3.5 stars
- **Maximum Calories**: ≤800 per serving
- **Minimum Reviews**: ≥5 reviews for reliability

## Implementation Details

### Key Classes and Methods

#### `RecipeDataPreprocessor`
Main preprocessing class with methods for each processing step:

- `load_data()`: Load raw CSV files
- `clean_basic_fields()`: Basic data cleaning
- `parse_time_fields()`: Time field processing
- `clean_nutrition_data()`: Nutrition data processing
- `process_ingredients()`: Ingredient processing
- `process_categories_and_keywords()`: Category processing
- `process_ratings_and_reviews()`: Review integration
- `create_final_features()`: Feature engineering
- `filter_for_healthy_recommendations()`: Quality filtering
- `save_processed_data()`: Save processed datasets

### Output Datasets

#### `processed_recipes_all.csv`
Complete processed dataset with all recipes and engineered features.

#### `processed_recipes_healthy.csv`
Filtered dataset containing only recipes meeting healthy recommendation criteria.

### Key Features in Final Dataset

#### Core Recipe Information
- `RecipeId`, `Name`, `Description`, `RecipeCategory`
- `TotalTimeMinutes`, `TimeCategory`
- `IngredientCount`, `ComplexityCategory`, `IngredientList`

#### Nutrition & Health
- `Calories`, `CalorieCategory`, `HealthinessScore`
- All original nutrition columns (FatContent, ProteinContent, etc.)
- `HasHealthyIngredients`, `HasUnhealthyIngredients`

#### Quality & Popularity
- `AvgRating`, `ReviewCount`, `PopularityScore`
- `RecommendationScore`

#### Dietary Restrictions
- `IsVegetarian`, `IsVegan`, `IsGlutenFree`, `IsLowCarb`

#### Meal Planning
- `MealType` (Quick Snack, Light Meal, etc.)

## Usage Instructions

### Running the Preprocessing Pipeline

```python
from src.data_preprocessing import RecipeDataPreprocessor

# Initialize preprocessor
preprocessor = RecipeDataPreprocessor()

# Load and process data
recipes_df, reviews_df = preprocessor.load_data('notebooks/recipes.csv', 'notebooks/reviews.csv')

# Apply all preprocessing steps
recipes_df = preprocessor.clean_basic_fields(recipes_df)
recipes_df = preprocessor.parse_time_fields(recipes_df)
recipes_df = preprocessor.clean_nutrition_data(recipes_df)
recipes_df = preprocessor.process_ingredients(recipes_df)
recipes_df = preprocessor.process_categories_and_keywords(recipes_df)
recipes_df = preprocessor.process_ratings_and_reviews(recipes_df, reviews_df)
recipes_df = preprocessor.create_final_features(recipes_df)

# Filter for healthy recommendations
healthy_recipes = preprocessor.filter_for_healthy_recommendations(recipes_df)

# Save processed data
preprocessor.save_processed_data(recipes_df, 'data/processed_recipes_all.csv')
preprocessor.save_processed_data(healthy_recipes, 'data/processed_recipes_healthy.csv')
```

### Command Line Usage

```bash
cd /path/to/meal_recommender
python src/data_preprocessing.py
```

## Customization Options

### Adjusting Health Criteria
Modify filtering parameters in `filter_for_healthy_recommendations()`:
- `min_healthiness_score`: Minimum health score threshold
- `min_rating`: Minimum user rating threshold  
- `max_calories`: Maximum calories per serving
- Minimum review count for reliability

### Adding Custom Health Keywords
Extend the healthy/unhealthy keyword lists in the constructor:
```python
self.healthy_keywords.extend(['your', 'custom', 'keywords'])
self.unhealthy_keywords.extend(['your', 'custom', 'keywords'])
```

### Modifying Scoring Weights
Adjust weights in the recommendation score calculation:
```python
df['RecommendationScore'] = (
    df['HealthinessScore'] * 0.4 +        # Health weight
    df['PopularityScore'] * 10 * 0.3 +    # Popularity weight  
    (100 - df['TotalTimeMinutes'].fillna(60) / 2) * 0.2 +  # Time weight
    df['HasHealthyIngredients'].astype(int) * 10 * 0.1     # Ingredient weight
)
```

## Performance Considerations

### Memory Management
- Process data in chunks for very large datasets
- Use appropriate data types (int32 vs int64, float32 vs float64)
- Consider using categorical data types for repeated string values

### Processing Time
- Expected processing time: 10-30 minutes for full dataset
- Most time-intensive steps: ingredient parsing and review aggregation
- Consider parallel processing for ingredient analysis

## Quality Assurance

### Data Validation Checks
- Verify no duplicate RecipeIds in final dataset
- Check for reasonable value ranges in nutrition columns
- Validate that all categorical variables have expected values
- Ensure no critical missing values in key recommendation features

### Output Verification
- Compare record counts before and after processing
- Verify healthiness score distribution is reasonable
- Check that filtering produces expected number of healthy recipes
- Validate that all engineered features are properly calculated

## Next Steps

After preprocessing, the cleaned data can be used for:

1. **Exploratory Data Analysis**: Analyze patterns in healthy vs unhealthy recipes
2. **Feature Selection**: Identify most important features for recommendations
3. **Model Training**: Train collaborative filtering, content-based, or hybrid recommendation models
4. **Evaluation**: Assess recommendation quality using appropriate metrics
5. **Deployment**: Integrate into production recommendation system

## Troubleshooting

### Common Issues

#### Memory Errors
- Reduce chunk size when processing large files
- Use more efficient data types
- Process subsets of data for development/testing

#### Parsing Errors
- Check for malformed ingredient lists or keywords
- Verify CSV file encoding (UTF-8 recommended)
- Handle edge cases in time duration parsing

#### Missing Dependencies
- Ensure all required packages are installed: pandas, numpy, ast, re
- Check Python version compatibility (3.7+ recommended)

### Performance Optimization
- Use vectorized operations instead of loops where possible
- Consider using Dask for very large datasets
- Profile code to identify bottlenecks
- Cache intermediate results for iterative development
