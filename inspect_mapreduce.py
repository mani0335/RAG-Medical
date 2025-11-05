import inspect
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
print('MapReduceDocumentsChain init signature:')
print(inspect.signature(MapReduceDocumentsChain))
print('\nClass doc:')
print(MapReduceDocumentsChain.__doc__[:1000])
try:
    fields = getattr(MapReduceDocumentsChain, '__fields__', None)
    print('\n__fields__ present:', bool(fields))
    if fields:
        for k,v in fields.items():
            print(k, '->', getattr(v, 'type_', repr(v)))
except Exception as e:
    print('Fields inspect failed:', e)
