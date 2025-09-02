# init repo and push to GitHub
git init
git add -A
git commit -m "v0.1.1"
git branch -M main
git remote add origin https://github.com/jlonghku/EcoHydroModel.git
git push -u origin main

# build package
python -m pip install --upgrade build
python -m build

# install locally
python -m pip install dist/ecohydromodel*.whl

# upload to TestPyPI
python -m pip install --upgrade twine
python -m twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple ecohydromodel

# upload to PyPI
python -m twine upload dist/*

# simple GitHub push
git add -A
git commit -m "update"
git push origin main

# release workflow: GitHub + PyPI
git add -A
git commit -m "release: v0.1.1"
git tag v0.1.1
git push origin main

rm -rf dist build *.egg-info
python -m build
python -m twine upload dist/*

# push all tags
git push --tags
git describe --tags

