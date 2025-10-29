from setuptools import setup


setup(
    name="seabass",
    setup_requires="setupmeta",
    versioning="dev",
    author="Karn Watcharasupat",
    entry_points={
        "console_scripts": [
            "seabass=seabass.__main__:main",
        ],
    },
)
