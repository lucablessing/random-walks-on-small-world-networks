# random-walks-on-small-world-networks

## Content

- [notebooks/](notebooks) contains the Jupyter notebook files, where all major work is done

  - The [Creation of the small world network.ipynb](notebooks/Creation%20of%20the%20small%20world%20network.ipynb) holds a description of the small world network creation process, as proposed in the ["Random Walks on Small World Networks" paper](https://arxiv.org/pdf/1707.02467.pdf), and the two random walk types used in this project
  
  - The [Final_report.ipynb](notebooks/Final_report.ipynb) includes the most relevant results
  
  - The sub-directories hold data produced during the work, e.g. plots and graphs
  
  - The other notebooks are used to seperately work on multiple topics and areas simultaneously

- [scripts/](scripts) contains different Python files with the methods for graph creation, random walks, etc.

## Code environment

### Notes on setup
- Github and Jupyter Notebooks
  - https://towardsdatascience.com/introduction-to-github-for-data-scientists-2cf8b9b25fba 
- Pipenv guide 
  - https://realpython.com/pipenv-guide/

### Setup using pipenv

1. In your Terminal (on Mac or Linux) go to the directory you want to work in
2. Install pipenv
1. Setup an environment or install from Pipfile 
2. Start shell 
3. Install ```nodeenv```
4. Exit and restart shell to apply changes
5. Install some basic libraries
6. Install widgets and you're good to go!

```
$ cd ~/PATH/TO/YOUR/DIRECTORY
$ pip3 install --user pipenv # install pipenv if you don't have it yet
$ pipenv install  # set up environment from Pipfile
$ pipenv shell    # activate environment
$ nodeenv -p      # install nodeenv in env for jupyter-widgets
$ exit            # deactivate env
$ pipenv shell    # activate env
$ pipenv install numpy scipy matplotlib jupyter jupyterlab
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
