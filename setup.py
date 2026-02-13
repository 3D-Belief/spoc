from setuptools import setup, find_packages

setup(
    name="spoc",
    version="0.1.0",
    description="SPOC",
    python_requires=">=3.9",
    packages=find_packages(
        where=".",
        include=[
            "embodied", "embodied.*",
            "spoc_utils", "spoc_utils.*",
            "environment", "environment.*",
        ],
    ),
    include_package_data=True,
)
