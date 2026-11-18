from setuptools import setup, find_packages

setup(
    name="iot-anomaly-detector",
    version="1.0.0",
    description="IoT network anomaly detection via Convolutional Autoencoder with Attention",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10,<3.12",  # 3.10/3.11 recommended; tensorflow-metal not fully ready for 3.12+
    install_requires=[
        "dpkt>=1.9.8",
        "pandas>=2.0.0",
        "numpy==1.24.3",           # pinned: tensorflow-macos 2.13.0 requires numpy<=1.24.3; numpy 2.x breaks TF 2.13 API
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0,<1.6",     # imbalanced-learn 0.12.x requires sklearn <1.6
        "imbalanced-learn==0.12.4",     # 0.13+ requires numpy>=1.25 which conflicts with tensorflow-macos 2.13
        "tensorflow==2.13.0",      # pinned: must match tensorflow-metal version
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "isort", "mypy"],
        "hpo": ["scikit-optimize>=0.9.0"],
        "adversarial": ["adversarial-robustness-toolbox>=1.16.0"],
        "explainability": ["shap>=0.42.0"],
    },
    entry_points={
        "console_scripts": [
            "iot-process-pcap=iot_anomaly.cli.process_pcap:main",
            "iot-train=iot_anomaly.cli.train:main",
        ]
    },
)
