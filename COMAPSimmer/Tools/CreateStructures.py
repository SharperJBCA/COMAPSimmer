import h5py
import sys
import pickle

cpyAttrs = {}
cpyTarget = {}

def copy_attrs_dict(name,obj):
    if isinstance(obj, h5py.Group):
        if not name in cpyTarget:
            if len(obj.attrs.keys()) > 0:
                cpyAttrs[name] = {}
                cpyAttrs[name]['attrs'] = {}

        for key, val in obj.attrs.items():
            print(name, key, val)
            if not key in cpyAttrs[name]['attrs']:
                cpyAttrs[name]['attrs'][key] = ''
    return None

def copy_dsets_dict(name,obj):
    if isinstance(obj, h5py.Dataset):
       # print(name)
        if not (name in cpyTarget):
            cpyTarget[name] = list(obj.shape) + [obj.dtype]
                    
    return None

cpySource = h5py.File(sys.argv[1],'r')
cpySource.visititems(copy_attrs_dict)
cpySource.visititems(copy_dsets_dict)

data = {'datasets':cpyTarget,
        'attrs': cpyAttrs}

with open('level1-struct.p','wb') as fp:
    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('level1-struct.p','rb') as fp:
    data = pickle.load(fp)

print(data)

#print(cpyTarget)
