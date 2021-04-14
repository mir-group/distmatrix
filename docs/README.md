# Dependencies

- Doxygen
- Sphinx w/extensions
```
pip install sphinx sphinx_rtd_theme breathe
```

# Make docs
```
cd docs
doxygen
make html
```
Then look at `_build/html/index.html`.
