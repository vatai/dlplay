#!/bin/bash

python -m vecto benchmark \
	analogy \
	--method LRCos \
	--path_out analogy/ \
	w2v/ep_010/ \
	BATS_3.0
