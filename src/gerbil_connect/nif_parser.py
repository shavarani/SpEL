from pynif import NIFCollection, NIFContext
#https://github.com/wetneb/pynif


class ParsedNIFContext:
    def __init__(self, beginIndex=None, endIndex=None, mention=None, sourceUrl=None, uri=None, is_hash_based_uri=False):
        self._context = NIFContext(beginIndex, endIndex, mention, sourceUrl, uri, is_hash_based_uri)

    @staticmethod
    def from_nif_context(nif_context):
        p = ParsedNIFContext()
        p._context = nif_context
        return p

    def add_phrase(self, beginIndex=None, endIndex=None, annotator=None, score=None, taIdentRef=None, taClassRef=None,
                   taMsClassRef=None, uri=None, source=None, is_hash_based_uri=False):
        self._context.add_phrase(
            beginIndex, endIndex, annotator, score, taIdentRef, taClassRef, taMsClassRef, uri, source, is_hash_based_uri)

    @property
    def request_id(self):
        assert "request" in self._context.uri
        u = self._context.uri.split("#")[0]
        return int(u.split("_")[-1])

    def __str__(self):
        return self._context.mention

    @property
    def mention(self):
        return self._context.mention

    @property
    def turtle(self):
        return self._context.turtle


class NIFParser:
    def __init__(self, nif_str, format='turtle', uri=None):
        self._collection = NIFCollection.loads(nif_str, format=format, uri=uri)

    def nif_str(self, format='turtle'):
        return self._collection.dumps(format=format)

    @property
    def contexts(self):
        return [ParsedNIFContext.from_nif_context(c) for c in self._collection.contexts]