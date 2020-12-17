# random-walks-on-small-world-networks

### Notes on setup
- Github and Jupyter Notebooks
  - https://towardsdatascience.com/introduction-to-github-for-data-scientists-2cf8b9b25fba 
- Pipenv guide 
  - https://realpython.com/pipenv-guide/

### Setup using pipenv

1. Install from Pipfile 
2. Start shell 
3. Install ```nodeenv```
4. Exit and restart shell to apply changes
5. Install widgets and you're good to go!

```
$ pipenv install  # set up environment from Pipfile
$ pipenv shell    # activate environment
$ nodeenv -p      # install nodeenv in env for jupyter-widgets
$ exit            # deactivate env
$ pipenv shell    # activate env
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
