# Extreme Adaptation for Personalized Neural Machine Translation

This repository contains the code for the paper [Extreme Adaptation for Personalized Neural Machine Translation](http://www.cs.cmu.edu/~pmichel1/hosting/extreme-adaptation-personalized.pdf).

## Data

The data used in the paper is the SATED dataset, available at this url: http://www.cs.cmu.edu/~pmichel1/sated/.

Additional experiments were performed on the gender annotated europarl corpus from \[1\], available at this url https://www.kaggle.com/ellarabi/europarl-annotated-for-speaker-gender-and-age.

You can download all the data by running:

```bash
# SATED
wget http://www.cs.cmu.edu/~pmichel1/hosting/sated-release-0.9.0.tar.gz
tar xvzf sated-release-0.9.0.tar.gz
# Europarl
wget https://www.kaggle.com/ellarabi/europarl-annotated-for-speaker-gender-and-age/downloads/europarl-annotated-for-speaker-gender-and-age.zip
unzip europarl-annotated-for-speaker-gender-and-age.zip
```

## Requirements

This project was coded in [Dynet](https://github.com/clab/dynet). It should be working with the `2.0.3` release which you can install by running:

```bash
pip install dynet==2.0.3
```

## References

If you use this code or the SATED dataset in you research, consider citing the original paper:

```
TBD
```

Other references:

\[1\]: https://aclanthology.coli.uni-saarland.de/pdf/E/E17/E17-1101.pdf
