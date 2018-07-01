#!/bin/bash

for f in $(git status --porcelain | grep '^??' | sed 's/^?? //'); do
    test -L "$f" && echo $f >> .gitignore; # add symlinks
done


