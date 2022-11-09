# `ASAP` - Critical `A`ssessment of `S`ynthetic `A`ccessibility scores in computer-assisted synthesis `P`lanning

This repository contains the source code for reproducing the results from the article, for preprint see .... 
Also, raw results are available

## What is ASAP?
> We assess if synthetic accessibility scores can reliably predict the outcomes of retrosynthesis planning.
> Using a specially prepared compounds database, we examine the outcomes of the retrosynthetic tool AiZynthFinder.
> We test whether synthetic accessibility scores: SAscore, SYBA, SCScore, and RAscore accurately predict the results of retrosynthesis planning.
> Furthermore, we investigate if synthetic accessibility scores can speed up retrosynthesis planning by better prioritizing explored partial synthetic routes and thus reducing the size of the search space.
> For that purpose, we analyze the AiZynthFinder partial solutions search trees, their structure, and complexity parameters, such as the number of nodes, or treewidth.

## Installation

Just run the command:


```sh
./run configure
```

This command clones the repositories of analyzed scores, creates separate conda environments conforming to their installation requirements and decompresses deposited raw results.
For more details, see the `run` file.

## Usage
1. Run AiZynthFinder for generating its search trees on the molecules database:
```sh
run main ai
```
Generated trees are then deposited in the `results/db.sqlite` file as an SQLite database.
2. For generated trees compute appropriate statistics:
```sh
run main ai_postprocess
```
3. Statistical analysis of search trees is available in notebooks: stats, stats_selected, and setups_table.
