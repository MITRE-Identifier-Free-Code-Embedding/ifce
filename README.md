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

* add total function count to diagrams / graphs
* change 
    coloration of tsne 
    figure size
* add error bars
* identify groups that models perform well on in signsrch experiments
* Identify functions that models perform well on

Discussions
* Validate signsrch experiments
* Work on slide materials


Open questions
* How to deal with presenting one sided statistical tests?
* Can OOD tests be improved? If so, how?


If VLLM problem persists, use Popen to run a vllm serve, and then use the query function to request expected objects. 

