# semantic free code embedding

## Overview

What is Semantic Free Code Embedding?

Semantic Free Code Embedding is an experiment suite designed to test and improve code embedding models at positioning their embeddings in space according to what a function does, and not how its variables are named.

This is accomplished by training with a contrastive learning objective that associates functions with their pseudo-c representations, instead of those with similar naming conventions or source representations.

Evaluation is accomplished through the following benchmarks:

* Assemblage - source to decomplilation association (and vice versa)
* Signsrch - "magic" constant association

## TODO

Software
* Consolidate similar functionality across files
* Segment processing from visualization in some files
* Observe Quantization effects
* prevent algorithms with one constant from being evaluated


Discussions
* Validate signsrch experiments
* Work on slide materials
* For signsrch exp 2, some algorithm groups have more constants than others. This makes them generate more AP scores and weights summaries stats towards their performance v. equal weighting. In reality these should be aggregations of MAPs?


Open questions
* How to deal with presenting one sided statistical tests?
* Can OOD tests be improved? If so, how?


If VLLM problem persists, use Popen to run a vllm serve, and then use the query function to request expected objects. 

