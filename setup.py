from setuptools import setup

setup(
    name="is_project",
    version="0.0.1",
    description="",
    url="",
    author="miquely",
    license="MIT",
    packages=["is_project", "is_project.conf"],
    package_dir={"": "."},
    entry_points={
        "console_scripts": [
            "is-project=is_project.main:main",
        ],
    },
    zip_safe=False,
    install_requires=[
        "is-msgs==1.1.18",
        "is-wire==1.2.1",
        "numpy==1.24.4",
        "opencv-python==4.8.0.74",
        "six",
        "imutils",
    ],
)

# $ python3 -m venv .venv
# $ which python3
# /usr/bin/python3
# $ source .venv/bin/activate
# $ which python3
# /home/miquelly/Desktop/is_project/.venv/bin/python3
