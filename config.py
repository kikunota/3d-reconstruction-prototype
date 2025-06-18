"""
Configuration settings for 3D Reconstruction application
"""
import os

class Config:
    """Base configuration"""
    
    # File upload settings
    MAX_CONTENT_LENGTH = 512 * 1024 * 1024  # 512MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    
    # Processing settings
    MAX_IMAGES = 500
    MAX_IMAGE_SIZE = 1600  # Max dimension for processing
    DEFAULT_FEATURE_TYPE = 'SIFT'
    
    # Bundle adjustment settings
    DEFAULT_OUTLIER_THRESHOLD = 2.0
    DEFAULT_MAX_ITERATIONS = 50
    DEFAULT_REFINEMENT_ITERATIONS = 2
    
    # Security settings
    UPLOAD_FOLDER = 'uploads'
    DOWNLOAD_RETENTION_HOURS = 1
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = 'development'
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

# Default configuration
config = DevelopmentConfig()