"""
The SpEL annotation visualization script. You can use this script as a playground to explore the capabilities and
limitations of the SpEL framework.
"""
import torch
from spel.model import SpELAnnotator
from spel.data_loader import dl_sa
from spel.utils import chunk_annotate_and_merge_to_phrase
from spel.candidate_manager import CandidateManager
import streamlit as st
from annotated_text import annotated_text
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model():
    load_aida_finetuned = True
    load_full_vocabulary=True
    candidate_setting = "n"
    model = SpELAnnotator()
    model.init_model_from_scratch(device=device)
    candidates_manager_to_use = CandidateManager(dl_sa.mentions_vocab,
                                                 is_kb_yago=candidate_setting == "k",
                                                 is_ppr_for_ned=candidate_setting.startswith("p"),
                                                 is_context_agnostic=candidate_setting == "pg",
                                                 is_indexed_for_spans=True) if candidate_setting != "n" else None
    if load_aida_finetuned and not load_full_vocabulary:
        model.shrink_classification_head_to_aida(device=device)
        model.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=3)
    elif load_aida_finetuned:
        model.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=4)
    else:
        model.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=2)
    return model, candidates_manager_to_use

annotator, candidates_manager = load_model()
st.title("SpEL Prediction Visualization")
st.caption('Running the \"[SpEL-base-step3-500K.pt](https://vault.sfu.ca/index.php/s/8nw5fFXdz2yBP5z/download)\" model without consideration of any hand-crafted candidate sets. For more information please checkout [SpEL\'s github repository](https://github.com/shavarani/SpEL).')
mention = st.text_input("Enter the text:")
process_button = st.button("Annotate")

if process_button and mention:
    phrase_annotations = chunk_annotate_and_merge_to_phrase(
        annotator, mention, k_for_top_k_to_keep=5, normalize_for_chinese_characters=True)
    last_step_annotations = [[p.words[0].token_offsets[0][1][0],
                              p.words[-1].token_offsets[-1][1][-1],
                              (dl_sa.mentions_itos[p.resolved_annotation], p.subword_annotations)]
                             for p in phrase_annotations if p.resolved_annotation != 0]
    if candidates_manager:
        for p in phrase_annotations:
            candidates_manager.modify_phrase_annotation_using_candidates(p, mention)
    if last_step_annotations:
        anns = sorted([(l_ann[0], l_ann[1], l_ann[2][0]) for l_ann in last_step_annotations], key=lambda x: x[0])
        begin = 0
        last_char = len(mention)
        anns_pointer = 0
        processed_anns = []
        anno_text = []
        while begin < last_char:
            if anns_pointer == len(anns):
                processed_anns.append((begin, last_char, "O"))
                anno_text.append(mention[begin: last_char])
                begin = last_char
                continue
            first_unprocessed_annotation = anns[anns_pointer]
            if first_unprocessed_annotation[0] > begin:
                processed_anns.append((begin, first_unprocessed_annotation[0], "O"))
                anno_text.append(mention[begin: first_unprocessed_annotation[0]])
                begin = first_unprocessed_annotation[0]
            else:
                processed_anns.append(first_unprocessed_annotation)
                anns_pointer += 1
                begin = first_unprocessed_annotation[1]
                anno_text.append((mention[first_unprocessed_annotation[0]: first_unprocessed_annotation[1]], first_unprocessed_annotation[2]))
        annotated_text(anno_text)
    else:
        annotated_text(mention)
