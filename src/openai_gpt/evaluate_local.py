import time
import logging
from spel.utils import postprocess_annotations
from spel.data_loader import dl_sa
from spel.configuration import get_checkpoints_dir, get_logdir_dir, get_exec_run_file, get_aida_train_canonical_redirects
from openai_gpt.utils import chunk_and_annotate

logging.basicConfig(filename='openai_gpt.log', filemode='w', level=logging.DEBUG)

FINAL_ANNOTATION_POSTPROCESSING_ALLOWED = True
dl_sa.shrink_vocab_to_aida()
mentions_dictionary = dict({(k.lower().replace("_", " "), k) for k, v in dl_sa.mentions_vocab.items()})


def convert_mention(m):
    m = m.lower()
    if m in ["u.s.", "us"]:
        return "United_States"
    elif m == "uk":
        return "United_Kingdom"
    elif m == "eu":
        return "European_Parliament"
    elif m == "china":
        return "People's_Republic_of_China"
    r = mentions_dictionary[m] if m in mentions_dictionary else ''
    if r == '' and m not in ["sport", "country", "person", "city", "team", "group", "organization", "location", "event",
                             "ethnicgroup", "nationality", "date", "technology", "services", "product", "league", "age",
                             "medicalcondition", "politicalparty", "profession", "material", "soccer", "rugby",
                             "tennis", "basketball", "golf", "cricket", "badminton", "squash", "baseball",
                             "ski jumping", "freestyle skiing", "alpine skiing", "speed skating", "boxing"]:
        potentials = [y for y in mentions_dictionary if m in y]
        # for the case of multiple match we have no choice but to choose one of the options. We select the first match!
        r = mentions_dictionary[potentials[0]] if len(potentials) else ''
    return r


class GPTAnnotator:
    def __init__(self):
        super(GPTAnnotator, self).__init__()
        self.checkpoints_root = get_checkpoints_dir()
        self.logdir = get_logdir_dir()
        self.exec_run_file = get_exec_run_file()

    def annotate(self, nif_collection, **kwargs):
        assert len(nif_collection.contexts) == 1
        context = nif_collection.contexts[0]
        time.sleep(3)
        kb_prefix = 'http://en.wikipedia.org/wiki/'
        last_step_annotations = [[p[0], p[1], (convert_mention(p[2]), None)] for p in chunk_and_annotate(context.mention)]
        logging.info('converted the extracted mentions:')
        logging.info('='*80)
        logging.info(str([x for x in last_step_annotations if x[2][0]]))
        logging.info('='*80)
        if FINAL_ANNOTATION_POSTPROCESSING_ALLOWED:
            last_step_annotations = postprocess_annotations(last_step_annotations, context.mention)

        canonical_redirects = self.get_canonical_redirects()

        for l_ann in [(l_ann[0], l_ann[1], (
                canonical_redirects[l_ann[2][0]], l_ann[2][1]) if l_ann[2][0] in canonical_redirects else l_ann[2])
                      for l_ann in last_step_annotations]:
            if not l_ann[2][0]:
                continue
            try:
                kbp = kb_prefix[l_ann[2][0]] if type(kb_prefix) == dict else kb_prefix
            except KeyError:
                kbp = kb_prefix['[defalt_prefix]']
            context.add_phrase(
                beginIndex=l_ann[0],
                endIndex=l_ann[1],
                annotator='http://sfu.ca/openai_gpt/annotator',
                taIdentRef=kbp+l_ann[2][0].replace("\"", "%22"))

    @staticmethod
    def get_canonical_redirects():
        return get_aida_train_canonical_redirects()

    def init_model_from_scratch(self, device):
        return

    def shrink_classification_head_to_aida(self, device):
        return

    def load_pretrained_model_checkpoint(self, device, load_finetuned, load_retokenized_wikipedia_finetuned):
        return

    def load_checkpoint(self, checkpoint_name, device="cpu", rank=0, load_from_torch_hub=False, finetuned_after_step=1):
        return
