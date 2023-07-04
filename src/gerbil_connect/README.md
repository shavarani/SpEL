# GERBIL connect

The GERBIL platform ([Röder et al., 2018](https://www.semantic-web-journal.net/system/files/swj1671.pdf)) 
is an evaluation toolkit (publicly available online) that eliminates any mistakes and allows for a fair comparison 
between methods. However, GERBIL is a Java toolkit, while most of modern entity linking  work is done in Python. 
GERBIL developers recommend using `SpotWrapNifWS4Test` (a middleware tool written in Java) to connect Python entity 
linkers to GERBIL. Because of the complexity of this setup, we have not been able to directly evaluate some of the 
earlier publications due to software version mismatches and communication errors between Python and Java. This is a 
drawback that discourages researchers from using GERBIL. To address this issue, in this package, we provide a Python 
equivalent of `SpotWrapNifWS4Test`, in Python, to encourage entity linking researchers to use GERBIL for fair 
repeatable comparisons.

---

## How to integrate `gerbil_connect` into your code?
The provided template has two python source code files:
1. `nif_parser.py` which provides the functionality to read the communicated messages from GERBIL and to write the annotated spans back to GERBIL, this file must not be changed and will be directly used by the integrated server.

2. `server_template.py` provides a template server in which you can import your implemented entity linking model and evaluate it using GERBIL (Röder et al., 2018). All the communication parts are worked out, and you will only need to replace `mock_entity_linking_model` with your model and load up its required resources in `generic_annotate` method. 

The provided template supports both Wikipedia in-domain test sets (i.e. AIDA-CoNLL) and out-of-domain test sets (e.g. KORE).

At the end, the annotation results will be stored in `annotate_{annotator_name}_result.json`.

For a completed example, please see `server.py` in `spel` package.

## how to evaluate using GERBIL?

1. Checkout [GERBIL repository](https://github.com/dice-group/gerbil) and run `cd gerbil/ && ./start.sh`
   - It will require Java 8 to run.
2. Once gerbil is running, run `python server_template.py` with your modifications which replaces `mock_entity_linking_model` with your entity linker. It will start listening on `http://localhost:3002/`.
3. Open a browser and type in `http://localhost:1234/gerbil/config`, this will open up the visual experiment configuration page of GERBIL.
4. Leave `Experiment Type` as `A2KB`, for `Matching` choose `Ma - strong annotation match`, and for `Annotator` set a preferred name (e.g. `Experiment 1`) and in `URI` set `http://localhost:3002/annotate_aida`.
   - You can also set the `URI` to `http://localhost:3002/annotate_wiki` for `MSNBC` dataset, to `http://localhost:3002/annotate_dbpedia` for `OKE`, `KORE`, and `Derczynski` datasets, and to `http://localhost:3002/annotate_n3` for `N3` Evaluation datasets.
5. Choose your evaluation `Dataset`s, for example choose `AIDA/CoNLL-Test A` and `AIDA/CoNLL-Test B` for evaluation on AIDA-CoNLL.
6. Check the disclaimer checkbox and hit `Run Experiment`.
7. Let GERBIL send in the evaluation documents (from the datasets you selected) one by one to the running server. Once it is done you can click on the URL printed at the bottom of the page (normally of the format `http://localhost:1234/gerbil/experiment?id=YYYYMMDDHHMM`) to see your evaluation results. 