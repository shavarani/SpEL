import os
import re
import backoff
from bs4 import BeautifulSoup
import openai
import tiktoken
import spacy
import logging

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    print(f"WARNING! you need to set environment variable OPENAI_API_KEY to your personal key before running this code,"
          f"\notherwise you may be locked out by openai!")
else:
    print(f"OpenAI API Key retreived from environment variables: {openai.api_key}")


spacy_model = spacy.load("en_core_web_sm")
PROMPT = "Annotate the wikipedia entities in the following paragraph, and " \
         "produce the output in html markup using the <mark> element and the data-entity attribute:\n\n"


def num_tokens_from_messages(messages, model_name="gpt-3.5-turbo-16k"):
    assert model_name == "gpt-3.5-turbo-16k", f"requested model name:{model_name} is not supported!"
    encoding = tiktoken.encoding_for_model(model_name)
    tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
    tokens_per_name = -1  # if there's a name, the role is omitted
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def create_messages(text, prompt=PROMPT):
    return [
        {"role": "system", "content": "You are a Wikipedia annotator."},
        {"role": "user", "content": prompt + text},
    ]


@backoff.on_exception(
    backoff.expo,
    openai.error.RateLimitError,
    max_time=240
)
def chat_completion(text, prompt=PROMPT, model_name="gpt-3.5-turbo-16k"):
    return openai.ChatCompletion.create(
        model=model_name,
        temperature=0,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        messages=create_messages(text, prompt),
        # max_tokens=4097
    )


def make_chat_completion_query(text, prompt=PROMPT, model_name="gpt-3.5-turbo-16k"):
    r = chat_completion(text, prompt, model_name)
    annotated = r.choices[0].message['content'].strip()
    if r.choices[0].finish_reason == "length":
        logging.warning("Potentially longer than maximum token, request returned with finish_reason == length")
    return annotated


def chunk_and_annotate(context_mention, chunk_size_limit=996):
    logging.info('Running chunk_and_annotate on input:')
    logging.info('='*80)
    logging.info(context_mention)
    logging.info('='*80)
    with spacy_model.select_pipes(enable=['tok2vec', "parser", "senter"]):
        doc = spacy_model(context_mention)
    result = ""
    chunk = ""
    sentences = [str(sentence) for sentence in doc.sents]
    final_text = " ".join(sentences)
    if final_text != context_mention:
        logging.warning("=" * 80)
        logging.warning("Potentially mis-aligned sentence spliting with spacy")
        logging.warning(">>> Sequence 1:")
        logging.warning(final_text)
        logging.warning(">>> Sequence 2:")
        logging.warning(context_mention)
        logging.warning("=" * 80)
    sent_index = 0
    while sent_index < len(sentences):
        while num_tokens_from_messages(create_messages(chunk)) < chunk_size_limit and sent_index < len(sentences):
            chunk += sentences[sent_index]
            sent_index += 1
        logging.info(f"making a query with size: {num_tokens_from_messages(create_messages(chunk))} tokens!")
        result += make_chat_completion_query(chunk)
        chunk = ""
    result = '<p>' + result + '</p>'
    logging.info('received the final result from openai gpt model:')
    logging.info('='*80)
    logging.info(result)
    logging.info('='*80)
    soup = BeautifulSoup(result, 'html.parser')
    markups = soup.find_all('mark')
    final_results = []
    current_index = 0

    for markup in markups:
        try:
            entity = markup['data-entity']
        except KeyError:
            continue
        try:
            pattern = re.escape(markup.string)
        except TypeError:
            continue
        match = re.search(pattern, context_mention[current_index:])
        if match:
            start = current_index + match.start()
            end = current_index + match.end()
            current_index = end
            final_results.append((start, end, entity))
    logging.info('parsed gpt output and extracted spans:')
    logging.info('='*80)
    logging.info(str(final_results))
    logging.info('='*80)
    return final_results
