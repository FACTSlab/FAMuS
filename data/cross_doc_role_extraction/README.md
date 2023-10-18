Each instance in the cross_doc_role_extraction task contains the following fields:

1. `instance_id`: This is the unique id for this instance
2. `instance_id_raw_lome_predictor`: This is a unique id as used by the LOME's FrameNet parser when launching the annotation protocols.
3. `frame`: This is the gold frame for which roles are annotated on this instance.
4. `report_dict`: This contains the information of the Report Text. This has the following fields:

   - `doctext`: The raw report text.
   - `doctext-tok`: Gold tokens for the raw report text.
   - `all-spans`: A list of all span-tuples in the report text that act as the candidates for role fillers in the text. These candidates were provided to the annotators while selecting role fillers for each role for the given frame.
     Each span-tuple is a 6-tuple with the following values:
     _textual_span_, _char_start_idx_, _char_end_idx_, _token_start_idx_, _token_end_idx_
     An example of a tuple: `['abandoned', 125, 133, 21, 21, '']`
   - `frame-trigger-span`: The span-tuple corresponding to the frame's trigger in the report text.` indicating the textual_span, the char_start_idx, char_end_idx, token_start_idx, token_end_idx respectively.)
   - `role_annotations`: This is a dictionary that contains the role fillers from the Report Text for each role of the frame.

5. `source_dict`: This contains the information of the Source Text, which is a valid source (i.e. the trigger event in the report is mentioned in the source), for the report in this instance. Note that the report trigger is not necessarily present in the source. The same event could be described in other words. This dict contains the following fields:

   - `doctext`: The raw source text.
   - `doctext-tok`: Gold tokens for the raw source text.
   - `all-spans`: A list of all span-tuples in the source text that act as the candidates for role fillers in the text. These candidates were provided to the annotators while selecting role fillers for each role for the given frame.
     Each span-tuple is a 6-tuple with the following values:
     _textual_span_, _char_start_idx_, _char_end_idx_, _token_start_idx_, _token_end_idx_
   - `role_annotations`: This is a dictionary that contains the role fillers from the Source Text for each role of the frame.

6. `bool_platinum`: `True` if the role annotations are platinum i.e. they were re-annotated by expert annotators.
