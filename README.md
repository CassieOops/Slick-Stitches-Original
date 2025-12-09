SlickStiches
A machine learning fashion outfit classifier that categorizes outfits as "trendy" or "not trendy" using Support Vector Machine (SVM) with CDN/Cloud support.

Features
Image Classification: Classifies fashion outfit images with 85%+ accuracy
CDN Support: Classify images directly from URLs (CDN, e-commerce sites, etc.)
Cloud Training: Train from Google Cloud Storage or local files
Flexible CLI: Multiple modes for different use cases
Usage
Classify Images
# Classify from URL (CDN support)
python3 main.py --url "https://example.com/outfit.jpg"

# Classify multiple URLs
python3 main.py --urls "url1" "url2" "url3"

# Classify local image
python3 main.py --local ./path/to/image.jpg

# Classify all test images
python3 main.py --batch
Train Model
# Train from local images
python3 main.py --train

# Train from cloud URLs (GCS, CDN)
python3 main.py --train-from-cloud cloud_images_config.json
Technical Details
Algorithm: Support Vector Machine (SVM) with GridSearchCV
Image Processing: 15x15 pixel preprocessing with scikit-image
Accuracy: ~85% on test dataset
Categories: Trendy / Not Trendy
Requirements
Python 3.x
scikit-learn
scikit-image
numpy
Setup
See GCS_SETUP_GUIDE.md for cloud storage configuration.

About
No description, website, or topics provided.
Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Releases
No releases published
Create a new release
Packages
No packages published
Publish your first package
Contributors
2
@JTCSnaps
JTCSnaps Jacob Cordano
@joseph-shur
joseph-shur
Languages
Python
100.0%
Suggested workflows
Based on your tech stack
Publish Python Package logo
Publish Python Package
Publish a Python Package to PyPI on release.
Python application logo
Python application
Create and test a Python application.
SLSA Generic generator logo
SLSA Generic generator
Generate SLSA3 provenance for your existing release workflows
More workflows
Footer
