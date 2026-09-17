# Python binding-surface rules

Type: grilling
Status: open
Blocked by: 03

## Question

State the Python binding-surface rules precisely: which names and signatures are frozen (module creation by type name, `add_child_raw` child names, param names `weight/bias/scales/zeros`, the `linear.py` public API `get_weight_plan`/`prepare_weight`/`get_exec_plan`/`tune`/`forward_moe`), what internal reshape is allowed in `builders/_base.py` and `weight_format.py`, and the mechanical-adaptation recipe per builder (`linear`, `attention`, `ffn`, `moe`, `deltanet`, `mla`, `text_model`). Output: the compatibility section for the spec.
