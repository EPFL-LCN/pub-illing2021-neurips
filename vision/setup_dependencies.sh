#!/usr/bin/env bash
echo "Make sure conda is installed."
echo "Installing environment:"
conda env create -f env.yml || conda env update -f env.yml || exit
conda activate clappvision
