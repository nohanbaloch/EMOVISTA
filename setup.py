from setuptools import setup, find_packages

setup(
    name='emotion-aware-ai',
    version='0.1.0',
    description='Emotion-Aware AI Assistant - FER + Speech + Text fusion',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'keras',
        'librosa',
        'joblib',
        'customtkinter',
        'opencv-python',
        'sounddevice',
        'tqdm',
        'matplotlib',
        'pytest',
    ],
)
