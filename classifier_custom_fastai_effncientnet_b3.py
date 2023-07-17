from fastai.vision.all import *
import timm
from fastai.distributed import *
import warnings

warnings.simplefilter("ignore")

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

random_seed(42, True)


def create_cls_module(nf, n_out, lin_ftrs=None, ps=0.5, use_bn=False, first_bn=False, bn_final=False, lin_first=False, y_range=None):
    "Creates classification layer which takes nf flatten features and outputs n_out logits"
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    bns = [first_bn] + [use_bn]*len(lin_ftrs[1:])
    ps = L(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = []
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni,no,bn,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], bns, ps, actns):
        layers += LinBnDrop(ni, no, bn=bn, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if y_range is not None: layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)



path = r'Optimal_31_splitted'
bs = 16
sz = 224

data = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y= parent_label,
    splitter= GrandparentSplitter(train_name = "train",valid_name = "val"),
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(mult=1.0, do_flip=True, flip_vert=True, max_rotate=120.0, min_zoom=1.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75, xtra_tfms=None, size=224, mode='bilinear', pad_mode='reflection', align_corners=True, batch=False, min_scale=1.0))

dls = data.dataloaders(path, bs=bs)
dls.show_batch()

print("Number of classes: ",dls.c)

arch = "efficientnet_b3"
model = timm.create_model(arch, pretrained=True, in_chans=3, num_classes=0)
model.classifier = nn.Sequential(create_cls_module(nf=model.num_features, n_out=dls.c))


savedfilename = f'{path}_{arch}'

learn = Learner(dls, model, metrics=[accuracy,Precision(average='macro'), Recall(average='macro'), F1Score(average='macro')],  cbs=[CSVLogger(fname=savedfilename+'.csv', append=True),
                         SaveModelCallback(monitor='valid_loss', comp=None, min_delta=0.0,fname=savedfilename, every_epoch=False, with_opt=False, reset_on_fit=True)])#.to_fp16()

with learn.distrib_ctx(sync_bn=False):
    learn.fit_flat_cos(30, 3e-4)