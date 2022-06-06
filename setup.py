import setuptools

setuptools.setup(
    name="CKWS Adapted Refined Score Attack code",
    author="Marco Dijkslag",
    author_email="m.dijkslag@alumnus.utwente.nl",
    description="Code used in 'Passive query-recovery attack against secure conjunctive keyword search schemes'",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
)
