Implementation of

    Plank et al. Multilingual Part-of-Speech Tagging with Bidirectional Long-Short Term Memory Models and Auxiliary Loss. ACL 2016


Installation
---
Install polyglot dependencies
```
sudo apt-get install python-numpy libicu-dev
```

Install project dependencies
```
echo nnnlp_conda > .python-version
pyenv virtualenv miniconda3-latest nnnlp_conda
pip install -r requirements.txt
```

Download polyglot resources
```
polyglot download LANG:fi
polyglot download LANG:en
polyglot download LANG:de
polyglot download LANG:id
```


Run
---
Run main experiment
```
python code/run.py run_experiment results.csv
```

Run experiment on performance vs. dataset set
```
python code/run.py run_experiments_data_size results_data_size.csv
```