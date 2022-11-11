# Generate Contextualised Utterance Embeddings with BART
Set of python scripts to generate utterance embeddings representing utterances in the context of their parent paragraph. 
To use, run make_embeddings_file.py. In make_embeddings_file.py, change the model argument to specify the kind of 
contextualisation (options are aware (which generates embeddings representing utterances in the context of their parent 
paragraphs), naive (contextualised with a pretrained LM (BART)), and no_context (does not model context). This will populate
the appropriate directory in the context_embeddings directory.

Corpora must be formatted as corpora/Create LJ_paragraph_delimited.txt: new utterance = CODE|UTTERANCE/n, new paragraph=/n/n
