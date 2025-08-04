import json
import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import geopy.distance
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import glob

warnings.filterwarnings('ignore')

class EnhancedGeospatialTrainer:
    """
    Enhanced trainer with rich geospatial feature extraction and flexible file loading
    """
    
    def __init__(self, zone='south'):
        self.zone = zone
        self.base_path = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
        
        # Store loaded data
        self.points_df = None
        self.places_df = None
        self.pois_df = None
        self.waters_df = None
        self.landuse_df = None
        
        # Spatial indices for fast distance calculations
        self.places_tree = None
        self.pois_tree = None
        self.waters_tree = None
        self.landuse_tree = None # Added for landuse optimization
    
    def find_geojson_files(self, pattern):
        """Find all GeoJSON files matching a pattern"""
        search_patterns = [
            os.path.join(self.base_path, pattern),
            os.path.join(self.base_path, "**", pattern),  # Search in subdirectories
        ]
        
        files = []
        for search_pattern in search_patterns:
            files.extend(glob.glob(search_pattern, recursive=True))
        
        return files
    
    def load_geojson_file(self, filename):
        """Load and process a single GeoJSON file"""
        # Try exact path first
        file_path = os.path.join(self.base_path, filename)
        
        # If not found, search for the file
        if not os.path.exists(file_path):
            found_files = self.find_geojson_files(filename)
            if found_files:
                file_path = found_files[0]
                print(f"📍 Found {filename} at: {file_path}")
            else:
                print(f"⚠️  File not found: {filename}")
                return pd.DataFrame()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            data = []
            features_count = len(geojson_data.get('features', []))
            
            for feature in geojson_data.get('features', []):
                props = feature.get('properties', {})
                geom = feature.get('geometry', {})
                
                if not geom or 'type' not in geom:
                    continue
                
                if geom['type'] == 'Point':
                    if 'coordinates' not in geom or len(geom['coordinates']) < 2:
                        continue
                    lon, lat = geom['coordinates'][:2]
                    data.append({
                        'latitude': lat,
                        'longitude': lon,
                        'fclass': props.get('fclass', props.get('class', 'unknown')),
                        'name': props.get('name', ''),
                        'type': props.get('type', ''),
                        'amenity': props.get('amenity', ''),
                        'shop': props.get('shop', ''),
                        'landuse': props.get('landuse', ''),
                        'place': props.get('place', ''),
                        'natural': props.get('natural', ''),
                        'waterway': props.get('waterway', '')
                    })
                elif geom['type'] in ['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString']:
                    # For complex geometries, calculate centroid
                    try:
                        if geom['type'] == 'Polygon':
                            coords = geom['coordinates'][0]
                        elif geom['type'] == 'MultiPolygon':
                            coords = geom['coordinates'][0][0]
                        elif geom['type'] == 'LineString':
                            coords = geom['coordinates']
                        elif geom['type'] == 'MultiLineString':
                            coords = geom['coordinates'][0]
                        
                        # Calculate centroid
                        if coords and len(coords) > 0:
                            lons = [coord[0] for coord in coords if len(coord) >= 2]
                            lats = [coord[1] for coord in coords if len(coord) >= 2]
                            
                            if lons and lats:
                                centroid_lon = sum(lons) / len(lons)
                                centroid_lat = sum(lats) / len(lats)
                                
                                data.append({
                                    'latitude': centroid_lat,
                                    'longitude': centroid_lon,
                                    'fclass': props.get('fclass', props.get('class', 'unknown')),
                                    'name': props.get('name', ''),
                                    'type': props.get('type', ''),
                                    'amenity': props.get('amenity', ''),
                                    'landuse': props.get('landuse', ''),
                                    'place': props.get('place', ''),
                                    'natural': props.get('natural', ''),
                                    'waterway': props.get('waterway', ''),
                                    'geometry_type': geom['type']
                                })
                    except (IndexError, TypeError, KeyError) as e:
                        print(f"⚠️  Skipping invalid geometry in {filename}: {e}")
                        continue
            
            df = pd.DataFrame(data)
            print(f"✅ Loaded {filename}: {len(df)} features (from {features_count} total)")
            return df
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return pd.DataFrame()
    
    def load_all_geospatial_data(self):
        """Load all geospatial data files for the zone"""
        print(f"🌍 Loading geospatial data for {self.zone} zone...")
        print(f"📁 Base path: {self.base_path}")
        
        # Primary: Try zone-specific files (handle your actual file naming)
        zone_files = {
            'points': f'{self.zone}_points.geojson',
            'places': f'{self.zone}_places.geojson', 
            'pois': f'{self.zone}_pois.geojson',
            'waters': f'{self.zone}_waters.geojson',
            'landuse': f'{self.zone}_landuse.geojson'
        }
        
        # Load zone-specific files
        self.points_df = self.load_geojson_file(zone_files['points'])
        self.places_df = self.load_geojson_file(zone_files['places'])
        self.pois_df = self.load_geojson_file(zone_files['pois'])
        self.waters_df = self.load_geojson_file(zone_files['waters'])
        self.landuse_df = self.load_geojson_file(zone_files['landuse'])
        
        # Handle your specific file structure variations
        # Check for center vs centre naming
        if self.zone == 'center' and self.points_df.empty:
            self.points_df = self.load_geojson_file('centre_points.geojson')
        if self.zone == 'center' and self.places_df.empty:
            self.places_df = self.load_geojson_file('centre_places.geojson')
        if self.zone == 'center' and self.pois_df.empty:
            self.pois_df = self.load_geojson_file('centre_pois.geojson')
        if self.zone == 'center' and self.waters_df.empty:
            self.waters_df = self.load_geojson_file('centre_waters.geojson')
        if self.zone == 'center' and self.landuse_df.empty:
            self.landuse_df = self.load_geojson_file('centre_landuse.geojson')
        
        # Handle composite files like south_points_places.geojson and south_points_waters.geojson
        if self.zone == 'south':
            if self.places_df.empty:
                # Try south_points_places.geojson for places data
                composite_places = self.load_geojson_file('south_points_places.geojson')
                if not composite_places.empty:
                    # Filter for place-like features
                    place_features = composite_places[
                        (composite_places['place'].notna() & composite_places['place'].ne('')) |
                        (composite_places['fclass'].isin(['city', 'town', 'village', 'hamlet'])) |
                        (composite_places['name'].notna() & composite_places['name'].ne(''))
                    ]
                    if not place_features.empty:
                        self.places_df = place_features
                        print(f"✅ Extracted {len(place_features)} places from south_points_places.geojson")
            
            if self.waters_df.empty:
                # Try south_points_waters.geojson for water data
                composite_waters = self.load_geojson_file('south_points_waters.geojson')
                if not composite_waters.empty:
                    # Filter for water-like features
                    water_features = composite_waters[
                        (composite_waters['natural'].str.contains('water|lake|river', na=False)) |
                        (composite_waters['waterway'].notna() & composite_waters['waterway'].ne('')) |
                        (composite_waters['fclass'].str.contains('water|river|lake', na=False))
                    ]
                    if not water_features.empty:
                        self.waters_df = water_features
                        print(f"✅ Extracted {len(water_features)} water features from south_points_waters.geojson")
        
        # Fallback: Load any available geojson files if zone-specific ones are missing
        if self.points_df.empty:
            print(f"⚠️  No zone-specific points found. Searching for any point files...")
            # Search for point files in all zones
            point_files = [f'{zone}_points.geojson' for zone in ['north', 'south', 'east', 'west', 'center']]
            self.points_df = self.load_multiple_files_by_name(point_files)
        
        if self.places_df.empty:
            print(f"⚠️  No zone-specific places found. Searching for any place files...")
            # Search for place files in all zones + composite files
            place_files = [f'{zone}_places.geojson' for zone in ['north', 'south', 'east', 'west', 'center']]
            place_files.extend(['south_points_places.geojson'])  # Add composite files
            self.places_df = self.load_multiple_files_by_name(place_files)
        
        if self.pois_df.empty:
            print(f"⚠️  No zone-specific POIs found. Searching for any POI files...")
            poi_files = [f'{zone}_pois.geojson' for zone in ['north', 'south', 'east', 'west', 'center']]
            self.pois_df = self.load_multiple_files_by_name(poi_files)
        
        if self.waters_df.empty:
            print(f"⚠️  No zone-specific waters found. Searching for any water files...")
            water_files = [f'{zone}_waters.geojson' for zone in ['north', 'south', 'east', 'west', 'center']]
            water_files.extend(['south_points_waters.geojson'])  # Add composite files
            self.waters_df = self.load_multiple_files_by_name(water_files)
        
        if self.landuse_df.empty:
            print(f"⚠️  No zone-specific landuse found. Searching for any landuse files...")
            landuse_files = [f'{zone}_landuse.geojson' for zone in ['north', 'south', 'east', 'west', 'center']]  
            self.landuse_df = self.load_multiple_files_by_name(landuse_files)
        
        # Final fallback: Load any geojson files and categorize by content
        if self.points_df.empty:
            print(f"⚠️  Still no points found. Loading all geojson files...")
            self.auto_categorize_files()
        
        print(f"📊 Final Data Summary:")
        print(f"   Points: {len(self.points_df)}")
        print(f"   Places: {len(self.places_df)}")
        print(f"   POIs: {len(self.pois_df)}")
        print(f"   Waters: {len(self.waters_df)}")
        print(f"   Landuse: {len(self.landuse_df)}")
    
    def load_multiple_files_by_name(self, filenames):
        """Load and merge files by exact filenames"""
        all_data = []
        
        for filename in filenames:
            df = self.load_geojson_file(filename)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            # Remove duplicates based on coordinates
            merged_df = merged_df.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
            print(f"✅ Merged {len(all_data)} files: {len(merged_df)} unique features")
            return merged_df
        else:
            return pd.DataFrame()
    
    def auto_categorize_files(self):
        """Automatically categorize geojson files based on their content"""
        print("🔍 Auto-categorizing geojson files...")
        
        geojson_files = self.find_geojson_files('*.geojson')
        
        points_data = []
        places_data = []
        pois_data = []
        waters_data = []
        landuse_data = []
        
        for file_path in geojson_files:
            filename = os.path.basename(file_path)
            df = self.load_geojson_file(filename)
            
            if df.empty:
                continue
            
            # Categorize based on content
            has_place = df['place'].notna().any() and df['place'].ne('').any()
            has_amenity = df['amenity'].notna().any() and df['amenity'].ne('').any()
            has_landuse = df['landuse'].notna().any() and df['landuse'].ne('').any()
            has_natural_water = (df['natural'].str.contains('water|lake|river', na=False).any() or
                               df['waterway'].notna().any() and df['waterway'].ne('').any())
            
            if has_place:
                places_data.append(df)
                print(f"   📍 {filename} -> Places (has place data)")
            elif has_amenity:
                pois_data.append(df)
                print(f"   🏪 {filename} -> POIs (has amenity data)")
            elif has_landuse:
                landuse_data.append(df)
                print(f"   🏞️  {filename} -> Landuse (has landuse data)")
            elif has_natural_water:
                waters_data.append(df)
                print(f"   💧 {filename} -> Waters (has water features)")
            else:
                points_data.append(df)
                print(f"   📌 {filename} -> Points (default)")
        
        # Merge categorized data
        if points_data and self.points_df.empty:
            self.points_df = pd.concat(points_data, ignore_index=True).drop_duplicates(subset=['latitude', 'longitude'])
        if places_data and self.places_df.empty:
            self.places_df = pd.concat(places_data, ignore_index=True).drop_duplicates(subset=['latitude', 'longitude'])
        if pois_data and self.pois_df.empty:
            self.pois_df = pd.concat(pois_data, ignore_index=True).drop_duplicates(subset=['latitude', 'longitude'])
        if waters_data and self.waters_df.empty:
            self.waters_df = pd.concat(waters_data, ignore_index=True).drop_duplicates(subset=['latitude', 'longitude'])
        if landuse_data and self.landuse_df.empty:
            self.landuse_df = pd.concat(landuse_data, ignore_index=True).drop_duplicates(subset=['latitude', 'longitude'])

    def build_spatial_indices(self):
        """Build KDTree indices for fast spatial queries"""
        print("🔍 Building spatial indices...")
        
        if not self.places_df.empty:
            coords = np.array(list(zip(self.places_df['latitude'], self.places_df['longitude'])))
            self.places_tree = cKDTree(coords)
            print(f"   Places index: {len(coords)} points")
        
        if not self.pois_df.empty:
            coords = np.array(list(zip(self.pois_df['latitude'], self.pois_df['longitude'])))
            self.pois_tree = cKDTree(coords)
            print(f"   POIs index: {len(coords)} points")
        
        if not self.waters_df.empty:
            coords = np.array(list(zip(self.waters_df['latitude'], self.waters_df['longitude'])))
            self.waters_tree = cKDTree(coords)
            print(f"   Waters index: {len(coords)} points")

        # OPTIMIZATION: Added a KDTree for the landuse data
        if not self.landuse_df.empty:
            coords = np.array(list(zip(self.landuse_df['latitude'], self.landuse_df['longitude'])))
            self.landuse_tree = cKDTree(coords)
            print(f"   Landuse index: {len(coords)} points")
        
        print("✅ Spatial indices built")
    
    def calculate_distance_to_nearest(self, lat, lon, tree):
        """Calculate distance to nearest feature using KDTree"""
        if tree is None or tree.n == 0:
            return 999.0  # Large default distance
        
        query_point = np.array([[lat, lon]])
        distance, _ = tree.query(query_point, k=1)
        
        # Convert to kilometers (approximate, 1 degree ~ 111km)
        return distance[0] * 111.0
    
    def count_features_within_radius(self, lat, lon, tree, radius_km=0.5):
        """Count features within a radius using KDTree"""
        if tree is None or tree.n == 0:
            return 0
        
        query_point = np.array([lat, lon])
        radius_degrees = radius_km / 111.0  # Convert km to degrees (approximate)
        
        indices = tree.query_ball_point(query_point, radius_degrees)
        return len(indices)
    
    def get_landuse_at_point(self, lat, lon):
        """
        OPTIMIZED: Get landuse type at point using an efficient KDTree index.
        """
        if self.landuse_tree is None or self.landuse_df.empty:
            return 'unknown'

        query_point = np.array([[lat, lon]])
        # Query for the nearest neighbor (k=1)
        # distance is in degrees, index is the row number in the original coords array
        distance, index = self.landuse_tree.query(query_point, k=1)

        # Convert distance from degrees to km for the threshold check
        distance_km = distance[0] * 111.0

        # Only return landuse if point is reasonably close (within 1km)
        if distance_km <= 1.0:
            # Use the index to get the landuse feature from the dataframe
            landuse_feature = self.landuse_df.iloc[index[0]]
            return landuse_feature.get('fclass', landuse_feature.get('landuse', 'unknown'))
        else:
            return 'unknown'
            
    def extract_rich_features(self, df):
        """Extract rich geospatial features for all points"""
        print("🔍 Extracting rich geospatial features...")
        
        enhanced_features = []
        total_points = len(df)
        
        for idx, row in df.iterrows():
            if idx > 0 and idx % 1000 == 0:
                print(f"   Processing {idx}/{total_points} points...")
            
            lat, lon = row['latitude'], row['longitude']
            
            # Basic features
            features = {
                'latitude': lat,
                'longitude': lon,
                'fclass': row.get('fclass', 'unknown'),
                'name': row.get('name', ''),
            }
            
            # Distance-based features
            features['dist_to_nearest_place'] = self.calculate_distance_to_nearest(lat, lon, self.places_tree)
            features['dist_to_nearest_poi'] = self.calculate_distance_to_nearest(lat, lon, self.pois_tree)
            features['dist_to_nearest_water'] = self.calculate_distance_to_nearest(lat, lon, self.waters_tree)
            
            # Count-based features
            features['poi_count_500m'] = self.count_features_within_radius(lat, lon, self.pois_tree, 0.5)
            features['poi_count_1km'] = self.count_features_within_radius(lat, lon, self.pois_tree, 1.0)
            features['place_count_1km'] = self.count_features_within_radius(lat, lon, self.places_tree, 1.0)
            
            # Landuse feature (now optimized)
            features['landuse_type'] = self.get_landuse_at_point(lat, lon)
            
            # Legacy feature for compatibility
            features['near_settlement'] = features['dist_to_nearest_place'] <= 2.0
            
            enhanced_features.append(features)
        
        enhanced_df = pd.DataFrame(enhanced_features)
        print(f"✅ Feature extraction complete! Shape: {enhanced_df.shape}")
        
        return enhanced_df
    
    def generate_smart_labels(self, df):
        """Generate intelligent labels based on business logic"""
        print("🏷️  Generating smart labels...")
        
        labels = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Existing store locations (positive examples)
            if row['name'] in ['Big Bazaar', 'Reliance Trends', 'City Centre 1', "Spencer's"]:
                labels.append(1)
                continue
            
            # Known unsuitable locations
            if row['fclass'] == 'open_land' and row['poi_count_500m'] == 0:
                labels.append(0)
                continue
            
            # Business logic scoring
            # 1. Landuse scoring
            landuse = row.get('landuse_type', 'unknown')
            if landuse in ['residential', 'commercial', 'mixed', 'retail']:
                score += 3
            elif landuse in ['industrial']:
                score += 1
            elif landuse in ['forest', 'water', 'wetland', 'agricultural']:
                score -= 3
            
            # 2. POI density scoring
            poi_500m = row.get('poi_count_500m', 0)
            if poi_500m >= 8:
                score += 4
            elif poi_500m >= 5:
                score += 3
            elif poi_500m >= 2:
                score += 2
            elif poi_500m >= 1:
                score += 1
            
            # 3. Accessibility scoring
            dist_place = row.get('dist_to_nearest_place', 999)
            if dist_place <= 0.5:  # Very close to a place
                score += 3
            elif dist_place <= 1.0:
                score += 2
            elif dist_place <= 2.0:
                score += 1
            elif dist_place > 10.0:  # Very remote
                score -= 3
            
            # 4. POI proximity scoring
            dist_poi = row.get('dist_to_nearest_poi', 999)
            if dist_poi <= 0.2:  # Very close to POI
                score += 2
            elif dist_poi <= 0.5:
                score += 1
            elif dist_poi > 5.0:  # Very far from POIs
                score -= 2
            
            # 5. Water proximity (negative for too close, positive for moderate distance)
            dist_water = row.get('dist_to_nearest_water', 999)
            if dist_water <= 0.05:  # Too close to water
                score -= 3
            elif 0.1 <= dist_water <= 2.0:  # Good distance from water
                score += 1
            
            # 6. Place density in area
            place_count_1km = row.get('place_count_1km', 0)
            if place_count_1km >= 3:
                score += 2
            elif place_count_1km >= 1:
                score += 1
            
            # Convert score to binary label
            # Threshold: score >= 4 means suitable
            label = 1 if score >= 4 else 0
            labels.append(label)
        
        labels_series = pd.Series(labels, index=df.index)
        
        # Report label distribution
        positive_count = sum(labels)
        total_count = len(labels)
        
        print(f"✅ Labels generated:")
        if total_count > 0:
            print(f"   Suitable locations (1): {positive_count} ({positive_count/total_count*100:.1f}%)")
            print(f"   Not suitable (0): {total_count-positive_count} ({(total_count-positive_count)/total_count*100:.1f}%)")
        
        return labels_series
    
    def train_model(self, df, labels):
        """Train the machine learning model"""
        print("🧠 Training machine learning model...")
        
        # Prepare features for training
        feature_columns = [
            'latitude', 'longitude', 'fclass', 'landuse_type', 'near_settlement',
            'dist_to_nearest_place', 'dist_to_nearest_poi', 'dist_to_nearest_water',
            'poi_count_500m', 'poi_count_1km', 'place_count_1km'
        ]
        
        # Create training dataframe
        training_df = df[feature_columns].copy()
        
        # One-hot encode categorical features
        training_encoded = pd.get_dummies(
            training_df, 
            columns=['fclass', 'landuse_type'], 
            drop_first=False
        )
        
        # Prepare X and y
        X = training_encoded
        y = labels
        
        # Align X and y after potential NaN label removal
        valid_indices = y.notna()
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        print(f"Training data shape: {X.shape}")
        
        # Check if we have enough data
        if len(X) < 10:
            print("❌ Not enough valid training data. Need at least 10 samples.")
            return None, None
        
        # Check if we have both classes
        if len(y.unique()) < 2:
            print(f"❌ Need samples from both classes (0 and 1) for training. Found only: {y.unique()}")
            return None, None
        
        # Train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # If stratification fails (e.g., too few samples in one class), try without it
            print("⚠️ Stratification failed. Splitting without it.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model training complete!")
        print(f"   Accuracy: {accuracy:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, X.columns.tolist()
    
    def save_model(self, model, feature_columns):
        """Save the trained model and feature columns"""
        print("💾 Saving model and feature columns...")
        
        model_path = os.path.join(self.base_path, 'store_placement_model.joblib')
        columns_path = os.path.join(self.base_path, 'feature_columns.joblib')
        
        dump(model, model_path)
        dump(feature_columns, columns_path)
        
        print(f"✅ Model saved as: store_placement_model.joblib")
        print(f"✅ Feature columns saved as: feature_columns.joblib")

def main():
    """Main training function"""
    print("🚀 Starting Enhanced Store Placement Model Training")
    print("=" * 60)
    
    # You can change the zone here to match your files
    # Available zones: 'south', 'north', 'east', 'west', 'center'
    zone = 'south'  # Change this to test different zones
    
    # Initialize trainer
    trainer = EnhancedGeospatialTrainer(zone=zone)
    
    # Load all geospatial data
    trainer.load_all_geospatial_data()
    
    if trainer.points_df.empty:
        print("❌ No point data found. Please ensure at least one GeoJSON file with point data exists.")
        return False
    
    # Build spatial indices
    trainer.build_spatial_indices()
    
    # Extract rich features
    enhanced_df = trainer.extract_rich_features(trainer.points_df)
    
    # Generate labels
    labels = trainer.generate_smart_labels(enhanced_df)
    
    # Train model
    model, feature_columns = trainer.train_model(enhanced_df, labels)
    
    if model is None:
        print("❌ Model training failed.")
        return False
    
    # Save model
    trainer.save_model(model, feature_columns)
    
    print("\n🎉 Training completed successfully!")
    print("Generated files:")
    print("  - store_placement_model.joblib")
    print("  - feature_columns.joblib")
    
    return True

if __name__ == "__main__":
    main()