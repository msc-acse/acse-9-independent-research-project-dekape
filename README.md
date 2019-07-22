# FullwaveQC

This is a ReadMe file under construction. This page supports markdown. Access
[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.

Sample Makefile:

init:
    pip install -r requirements.txt

test:
    py.test tests

.PHONY: init test

compare to true data --> find observed file from fullwave, guarantees same shapes
check attr for models and data not created by fullwave. can access attr easily to change them

