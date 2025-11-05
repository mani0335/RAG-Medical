import inspect
candidates = [
    'langchain.chains.mapreduce',
    'langchain.chains.combine_documents.mapreduce',
    'langchain.chains.combine_documents.base',
    'langchain.chains.combine_documents',
]
for m in candidates:
    try:
        mod = __import__(m, fromlist=['*'])
        print('Module found:', m)
        for name in dir(mod):
            if 'MapReduce' in name or name.lower().startswith('mapreduce'):
                cls = getattr(mod, name)
                print('\nFound class:', name)
                try:
                    src = inspect.getsource(cls)
                except Exception as e:
                    src = f'could not get source: {e}'
                print('\n'.join(src.splitlines()[:60]))
    except Exception as e:
        print('Could not import', m, '->', e)
